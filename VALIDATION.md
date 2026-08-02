# Validation performed

The package was checked locally with:

1. Python byte-code compilation of every `.py` file.
2. TOML parsing and verification of the 11-class order.
3. JSON verification that all 11 classes have exactly 16 prompts.
4. Byte-for-byte comparison of the bundled `tri_medtsllm.py`, `tri_components.py`, and `classification.py` against the retrieved reference files.
5. Installation into a synthetic compatible repository, including model and dataset registry patching.
6. Header scanning and deterministic train/validation/test generation on 55 synthetic WFDB headers (five records for each class).
7. A lazy dataset item test with a mocked 5000-by-12 WFDB waveform, verifying an output shape of 512-by-12 and an integer target.

Not performed here:

- Downloading the 5.1 GB PhysioNet dataset.
- Downloading FLAN-T5-XL or MOMENT model weights.
- A full GPU training run.
