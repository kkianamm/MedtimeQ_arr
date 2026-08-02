# Source fidelity

The two combined-model files under `models/` are unmodified copies of the reference repository's current `main` branch:

- `models/tri_medtsllm.py`
- `models/tri_components.py`

The bundled `tasks/classification.py` is also an unmodified reference copy. It is optional during installation because an existing checkout may contain local changes.

The dataset-specific files are new:

- `datasets/ecg_arrhythmia.py`
- `scripts/prepare_ecg_arrhythmia.py`
- `configs/datasets/ecg_arrhythmia_combined.toml`
- `prompts/ecg_arrhythmia_rhythm.json`

Reference URLs:

- https://github.com/kkianamm/medtsllm-biomedcoop
- https://physionet.org/content/ecg-arrhythmia/1.0.0/

SHA-256 checksums of the bundled reference copies:

```text
e1dfe9e8b220a796497018b8bc9d5d70d2ba4551564e898e19c2fc4298696f54  models/tri_medtsllm.py
dfbd0ed3f691a3bb82feda3e5727fda1df81e6681289a99cdadf237c23a249b4  models/tri_components.py
1331cdd64ecf9b7659dc97b93f59aadebf5b5cf3a500fb0cdeb0fa0a3d7bd905  tasks/classification.py
```
