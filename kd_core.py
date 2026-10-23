"""Minimal GPT-BEF teacher-to-student regression distillation example."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Model


TEACHER_COLUMNS = ["t_out", "light", "temperature:drybulb", "occ", "Equip"]
STUDENT_COLUMNS = ["t_out", "temperature:drybulb"]
TARGET_COLUMNS = ["Energy_1", "Energy_2", "Energy_3", "Energy_4"]


class WindowDataset(Dataset):
    def __init__(self, teacher, student, targets, history, horizon):
        self.teacher, self.student, self.targets = teacher, student, targets
        self.history, self.horizon = history, horizon
        self.count = len(targets) - history - horizon + 1
        if self.count < 1:
            raise ValueError("Each chronological partition needs at least history + horizon rows")

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        end = index + self.history
        target_index = end + self.horizon - 1
        return (
            torch.from_numpy(self.teacher[index:end]),
            torch.from_numpy(self.student[index:end]),
            torch.from_numpy(self.targets[target_index]),
        )


class GPT2Regressor(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.gpt2 = GPT2Model.from_pretrained("gpt2")
        hidden = self.gpt2.config.n_embd
        self.feature_proj = torch.nn.Linear(input_dim, hidden)
        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(hidden, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 4),
        )

    def forward(self, features):
        projected = self.feature_proj(features)
        positions = torch.arange(features.shape[1], device=features.device).unsqueeze(0)
        # Preserve the archived main implementation: it adds wpe here, and
        # GPT2Model adds its own positional embedding to inputs_embeds again.
        embedded = projected + self.gpt2.wpe(positions)
        hidden = self.gpt2(inputs_embeds=embedded).last_hidden_state[:, -1, :]
        return self.regressor(hidden)


def load_data(path, history, horizon):
    frame = pd.read_csv(path)
    columns = list(dict.fromkeys(TEACHER_COLUMNS + TARGET_COLUMNS))
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError("Missing CSV columns: " + ", ".join(missing))
    if frame[columns].isna().any().any():
        raise ValueError("CSV contains missing feature or target values")
    if "Time" in frame:
        frame["Time"] = pd.to_datetime(frame["Time"])
        if not frame["Time"].is_monotonic_increasing:
            raise ValueError("Time must be in chronological order")

    split = len(frame) * 6 // 9
    if min(split, len(frame) - split) < history + horizon:
        raise ValueError("Train and test partitions both need history + horizon rows")

    # Fit on training rows only. The archived implementation fitted on all rows.
    scalers = {
        "teacher": StandardScaler().fit(frame.iloc[:split][TEACHER_COLUMNS]),
        "student": StandardScaler().fit(frame.iloc[:split][STUDENT_COLUMNS]),
        "target": StandardScaler().fit(frame.iloc[:split][TARGET_COLUMNS]),
    }
    teacher = scalers["teacher"].transform(frame[TEACHER_COLUMNS]).astype(np.float32)
    student = scalers["student"].transform(frame[STUDENT_COLUMNS]).astype(np.float32)
    targets = scalers["target"].transform(frame[TARGET_COLUMNS]).astype(np.float32)

    def partition(start, stop):
        return WindowDataset(
            teacher[start:stop], student[start:stop], targets[start:stop], history, horizon
        )

    return partition(0, split), partition(split, len(frame)), scalers


def run_epoch(model, loader, optimizer, device, teacher=None, alpha=0.5, beta=0.5):
    model.train()
    criterion = torch.nn.MSELoss()
    total = 0.0
    for teacher_x, student_x, target in loader:
        teacher_x = teacher_x.to(device)
        student_x = student_x.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        if teacher is None:
            loss = criterion(model(teacher_x), target)
        else:
            with torch.no_grad():
                teacher_output = teacher(teacher_x)
            student_output = model(student_x)
            loss = alpha * criterion(student_output, teacher_output)
            loss += beta * criterion(student_output, target)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


def predict_model(model, dataset, target_scaler, device, batch_size, use_teacher):
    model.eval()
    predictions, actual = [], []
    with torch.no_grad():
        for teacher_x, student_x, target in DataLoader(dataset, batch_size=batch_size):
            features = teacher_x if use_teacher else student_x
            predictions.append(model(features.to(device)).cpu().numpy())
            actual.append(target.numpy())
    predicted = target_scaler.inverse_transform(np.concatenate(predictions))
    observed = target_scaler.inverse_transform(np.concatenate(actual))
    return np.maximum(predicted, 0), observed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Chronological resultnew*.csv")
    parser.add_argument("--output", type=Path, default=Path("runs/kd"))
    parser.add_argument("--teacher-checkpoint", type=Path, help="Optional compatible teacher state_dict")
    parser.add_argument("--history", type=int, default=1008)
    parser.add_argument("--horizon", type=int, default=144)
    parser.add_argument("--teacher-epochs", type=int, default=10)
    parser.add_argument("--student-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.5)
    args = parser.parse_args()
    if min(args.history, args.horizon, args.batch_size) < 1:
        parser.error("history, horizon and batch-size must be positive")
    if args.alpha < 0 or args.beta < 0 or args.alpha + args.beta == 0:
        parser.error("alpha and beta must be nonnegative and not both zero")
    if (args.teacher_checkpoint is None and args.teacher_epochs < 1) or args.student_epochs < 1:
        parser.error("training epochs must be positive")

    train, test, scalers = load_data(args.data, args.history, args.horizon)
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output.mkdir(parents=True, exist_ok=True)

    teacher = GPT2Regressor(len(TEACHER_COLUMNS)).to(device)
    if args.teacher_checkpoint:
        teacher.load_state_dict(torch.load(args.teacher_checkpoint, map_location=device, weights_only=True))
    else:
        optimizer = torch.optim.AdamW(teacher.parameters(), lr=args.learning_rate,
                                      weight_decay=args.weight_decay)
        for epoch in range(args.teacher_epochs):
            loss = run_epoch(teacher, loader, optimizer, device)
            print("teacher epoch {}: {:.6f}".format(epoch + 1, loss))
        torch.save(teacher.state_dict(), args.output / "teacher.pt")
    teacher.eval()
    teacher.requires_grad_(False)

    student = GPT2Regressor(len(STUDENT_COLUMNS)).to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    for epoch in range(args.student_epochs):
        loss = run_epoch(student, loader, optimizer, device, teacher, args.alpha, args.beta)
        print("student epoch {}: {:.6f}".format(epoch + 1, loss))
    torch.save(student.state_dict(), args.output / "student.pt")

    for name, model, use_teacher in (("teacher", teacher, True), ("student", student, False)):
        predicted, actual = predict_model(model, test, scalers["target"], device,
                                          args.batch_size, use_teacher)
        pd.DataFrame({**{"{}_pred".format(col): predicted[:, i] for i, col in enumerate(TARGET_COLUMNS)},
                      **{"{}_true".format(col): actual[:, i] for i, col in enumerate(TARGET_COLUMNS)}}).to_csv(
                          args.output / "{}_predictions.csv".format(name), index=False)
    np.savez(args.output / "scalers.npz",
             teacher_mean=scalers["teacher"].mean_, teacher_scale=scalers["teacher"].scale_,
             student_mean=scalers["student"].mean_, student_scale=scalers["student"].scale_,
             target_mean=scalers["target"].mean_, target_scale=scalers["target"].scale_)
    (args.output / "run_config.json").write_text(json.dumps(vars(args), default=str, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
