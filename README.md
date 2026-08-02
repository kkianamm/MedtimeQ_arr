# Exact MedTsLLM + MOMENT + Q-Former + BioMedCoOp adaptation

This overlay runs the same combined model implementation from `kkianamm/medtsllm-biomedcoop` on PhysioNet's **A large scale 12-lead electrocardiogram database for arrhythmia study v1.0.0**.

## What is unchanged

The model pathway and its losses are unchanged:

`MedTsLLM tokens + MOMENT tokens -> gated fusion -> learned BLIP-2-style Q-Former -> LLM -> BioMedCoOp prototype classifier`

The package carries exact copies of `models/tri_medtsllm.py` and `models/tri_components.py`. The combined hyperparameters and loss weights match `configs/combined_ptbxl.toml` from the reference repository.

## Necessary dataset formulation

The PhysioNet headers can contain one rhythm diagnosis plus several additional conditions. The reference model and classification task expect **one integer label with cross-entropy**, not a multi-hot vector. Therefore, to keep the method unchanged, this overlay implements **single-label rhythm classification**:

- Use the eleven official rhythm labels: `SB, SR, AFIB, ST, AF, SA, SVT, AT, AVNRT, AVRT, SAAWR`.
- Treat the first `#Dx` code as the primary rhythm, as in the database headers, and keep the record only when no second rhythm code creates ambiguity.
- Permit any number of additional non-rhythm diagnoses, but do not use them as targets.
- Put only age and sex in the sample-specific text prompt. Diagnosis codes are deliberately excluded to prevent label leakage.

A full multi-label 63-condition task would require BCE/sigmoid targets and changes to the auxiliary losses and BioMedCoOp head; that would not be the exact same method.

## Files

- `datasets/ecg_arrhythmia.py`: lazy WFDB loader, canonical lead ordering, complete-record resampling to 512 points, train-stat normalization, labels, and age/sex descriptions.
- `scripts/prepare_ecg_arrhythmia.py`: scans all headers, filters unique-rhythm records, makes deterministic stratified 80/10/10 splits, and computes train-only normalization statistics.
- `prompts/ecg_arrhythmia_rhythm.json`: sixteen BioMedCoOp descriptions for each of the eleven rhythm classes.
- `configs/datasets/ecg_arrhythmia_combined.toml`: complete runnable configuration.
- `install.py`: copies the files and safely patches both model and dataset registries.
- `scripts/check_ecg_arrhythmia.py`: checks manifests, class coverage, normalization arrays, and optionally a waveform.

## Installation

Start with a clean checkout of the reference repository:

```bash
git clone https://github.com/kkianamm/medtsllm-biomedcoop.git
cd medtsllm-biomedcoop
pip install -r requirements.txt
pip install -r recommended.txt
```

From the extracted overlay directory:

```bash
python3 install.py /path/to/medtsllm-biomedcoop \
  --data-root /lambda/nfs/Kiana2/ecg-arrhythmia-1.0.0
```

The installer verifies the host BioMedCoOp and classification-task markers, copies the exact combined model files, and patches:

```python
# models/__init__.py
from .tri_medtsllm import TriMedTsLLM
model_lookup["tri_medtsllm"] = TriMedTsLLM

# datasets/__init__.py
from .ecg_arrhythmia import ecg_arrhythmia_datasets
dataset_lookup["ECG-Arrhythmia"] = ecg_arrhythmia_datasets
```

Existing registry files and overwritten combined-model files receive `.before_ecg_arrhythmia.bak` backups.

## Prepare the dataset

Inside the patched repository:

```bash
pip install -r requirements-ecg-arrhythmia.txt
python3 scripts/check_environment.py
# Only when the checker reports missing combined-model packages:
# pip install -r requirements-combined-model.txt

python3 scripts/prepare_ecg_arrhythmia.py \
  --root /lambda/nfs/Kiana2/ecg-arrhythmia-1.0.0
```

Expected input structure:

```text
ecg-arrhythmia-1.0.0/
├── ConditionNames_SNOMED-CT.csv
├── RECORDS
└── WFDBRecords/
    └── 01/010/JS00001.hea
                 JS00001.mat
```

Preparation creates:

```text
processed/rhythm_single/
├── splits/train.csv
├── splits/val.csv
├── splits/test.csv
├── normalization.npz
└── summary.json
```

The split is deterministic with seed 0. It is stratified independently within every rhythm class. Because the database contains one ECG per patient, record-level splitting is patient-level splitting for this dataset.

Check it:

```bash
python3 scripts/check_ecg_arrhythmia.py \
  --root /lambda/nfs/Kiana2/ecg-arrhythmia-1.0.0 \
  --read-waveform
```

## Train

```bash
RUN_ID="ecg_arrhythmia_combined_seed0"
mkdir -p outputs/results outputs/console

python3 -u train.py \
  configs/datasets/ecg_arrhythmia_combined.toml \
  "$RUN_ID" \
  2>&1 | tee "outputs/console/${RUN_ID}.log"
```

Or run:

```bash
bash scripts/train_ecg_arrhythmia.sh ecg_arrhythmia_combined_seed0
```

The reference classification task writes train/validation/test accuracy, macro F1, macro precision, and macro recall at each epoch to `outputs/results/<run_id>.json`.

## Memory note

The reference configuration uses batch size 16, FLAN-T5-XL, and MOMENT-1-base. The original repository requirements file does not list `momentfm`, although the combined model imports it. `requirements-combined-model.txt` lists that missing dependency separately because current MOMENT releases can require newer NumPy/Transformers than the older reference requirements file. Keep these values for a direct method match. If the GPU runs out of memory, reducing batch size changes only optimization dynamics, not the architecture, but it is no longer a hyperparameter-identical run.

## Optional high-resolution context

`moment_windows = 0` matches the reference PTB-XL loader, which supplies only `x_enc`. Setting it to a positive integer makes the adapter split each original 5000-sample ECG into that many context windows and pass `x_moment_windows`; use this only as a separate ablation.
