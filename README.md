# GPT-BEF knowledge distillation: core example

This folder shows the central teacher-to-student training path from *Distilling Temporal Knowledge: A GPT-BEF Based Framework for Building Energy Consumption Forecasting*. It is a small educational release, not the full experiment archive or a claim that the published numerical results can be reproduced exactly from this folder.

## What the code does

1. Reads one chronological CSV with five teacher inputs (`t_out`, `light`, `temperature:drybulb`, `occ`, `Equip`) and four targets (`Energy_1` to `Energy_4`).
2. Splits the rows 6/9 for training and 3/9 for testing. A sample uses 1,008 past rows to predict **one four-value target vector** at offset 144; it does not predict all 144 future rows.
3. Trains a GPT-2 regression teacher with five inputs, or loads a compatible teacher checkpoint.
4. Trains a two-input student (`t_out`, `temperature:drybulb`) against both the frozen teacher and the true target: `0.5 * MSE(student, teacher) + 0.5 * MSE(student, target)`.
5. Saves both models' predictions for comparison, plus the student weights and scaling parameters. Student inference uses only its two inputs.

Both models use the same GPT-2 backbone and regression head. The five-feature teacher is the GPT-BEF model used inside the archived KD experiment; the two-feature student is GPT-BEF+KD. A separate standalone full-feature experiment in `gpt2 test1.py` used a different regression head and is not implemented here. The student reduces required input features; it is **not** a smaller parameter-count model. The code follows the main archived output-matching KD path; attention-map distillation and other exploratory experiments are outside this example.

## Run

Use a Python environment with the packages in `requirements.txt` and access to the Hugging Face `gpt2` checkpoint. This release includes `data/resultnew1A.csv` as a representative 1A climate-zone dataset (9,072 chronological rows, 10 columns, no missing cells). From this folder:

```bash
python kd_core.py --data data/resultnew1A.csv --output runs/1A
```

To use a compatible, already trained five-input teacher state dictionary:

```bash
python kd_core.py --data data/resultnew1A.csv --teacher-checkpoint /path/to/teacher.pt --output runs/1A
```

The defaults are 10 teacher epochs, 5 student epochs, batch size 12, learning rate `1e-4`, weight decay `1e-3`, and a 1,008/144 history/horizon. Training a full GPT-2 backbone with a 1,008-row window is resource intensive. If memory is insufficient, reduce `--batch-size`. The output folder receives `student.pt`, `teacher_predictions.csv`, `student_predictions.csv`, `scalers.npz`, and `run_config.json`; it also receives `teacher.pt` if the teacher was trained in this run.

## Relationship to archived experiments

The main archived configuration is represented by `outputnew_*` in the original project. This cleaned example **fits scalers only on training rows** to prevent test-distribution leakage; the archived main script fitted scalers on the complete series before splitting. It also saves the final weights directly. Consequently, its outputs are not expected to match archived numbers exactly. The main archived loss is prediction-level and ground-truth MSE; this example does not claim to implement every method or supplemental diagnostic described in the paper.

The included CSV is an unchanged copy of the project's `data/resultnew1A.csv` (SHA-256 `dfa2f717d0e1f1f746543e63b90be61e08ffbd9784e8b19cf804a2b7fb699c37`). It is a simulation-derived tabular export used by the project, not an unprocessed EnergyPlus log. Other climate-zone CSVs, simulation configuration, published checkpoints, baselines, and post-publication experiment files are not included. Feature names are preserved as recorded; the mapping to physical sensors and the units of the four targets require the original data-generation metadata.

## Verification

The code was run in a temporary environment with Python 3.13.3, PyTorch 2.14.1, Transformers 5.20.2, NumPy 2.5.4, pandas 3.1.0, and scikit-learn 1.9.1. A short-window, one-epoch end-to-end check completed teacher training, student KD training, prediction, and file creation using a small GPT-2 configuration. Separately, both teacher and student initialized the actual pretrained `gpt2` weights and produced finite four-target outputs. The archived `outputnew_1A` teacher and student state dictionaries both loaded into these classes with zero missing and zero unexpected keys. The full 1,008-step, 10/5-epoch experiment was not rerun in this check.
