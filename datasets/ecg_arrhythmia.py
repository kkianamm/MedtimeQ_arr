"""PhysioNet ECG Arrhythmia Database adapter for the MedTsLLM-BioMedCoOp model.

This adapter intentionally formulates the dataset as *single-label rhythm
classification*.  The reference TriMedTsLLM/BioMedCoOp implementation uses one
integer target and cross-entropy; therefore a full multi-label formulation would
change the method.  Records are retained when exactly one of the eleven rhythm
SNOMED-CT codes is present.  Additional non-rhythm diagnoses are allowed but are
not used as targets or prompt text.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from scipy.signal import resample_poly

from .base import BaseDataset


RHYTHM_CLASSES = [
    ("SB", "Sinus Bradycardia", "426177001"),
    ("SR", "Sinus Rhythm", "426783006"),
    ("AFIB", "Atrial Fibrillation", "164889003"),
    ("ST", "Sinus Tachycardia", "427084000"),
    ("AF", "Atrial Flutter", "164890007"),
    ("SA", "Sinus Irregularity", "427393009"),
    ("SVT", "Supraventricular Tachycardia", "426761007"),
    ("AT", "Atrial Tachycardia", "713422000"),
    ("AVNRT", "Atrioventricular Node Reentrant Tachycardia", "233896004"),
    ("AVRT", "Atrioventricular Reentrant Tachycardia", "233897008"),
    ("SAAWR", "Sinus Atrium to Atrial Wandering Rhythm", "195101003"),
]
CLASS_CODES = [item[0] for item in RHYTHM_CLASSES]
CLASS_NAMES = [item[1] for item in RHYTHM_CLASSES]
SNOMED_TO_CLASS = {item[2]: item[0] for item in RHYTHM_CLASSES}
CLASS_TO_INDEX = {code: index for index, code in enumerate(CLASS_CODES)}
CANONICAL_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def _cfg_get(config: Any, key: str, default: Any) -> Any:
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def parse_header(header_path: str | Path) -> dict[str, Any]:
    """Parse waveform metadata and diagnosis codes from one WFDB header."""
    path = Path(header_path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        raise ValueError(f"Empty WFDB header: {path}")
    first = lines[0].split()
    if len(first) < 4:
        raise ValueError(f"Malformed first line in {path}: {lines[0]!r}")
    metadata: dict[str, Any] = {
        "record_name": first[0],
        "n_signals": int(first[1]),
        "sampling_rate": float(first[2].split("/")[0]),
        "signal_length": int(first[3]),
        "age": None,
        "sex": "unknown",
        "dx_codes": [],
    }
    for line in lines:
        if not line.startswith("#") or ":" not in line:
            continue
        key, value = line[1:].split(":", 1)
        key, value = key.strip().lower(), value.strip()
        if key == "age" and value.lower() not in {"", "nan", "na", "unknown"}:
            try:
                metadata["age"] = int(float(value))
            except ValueError:
                metadata["age"] = None
        elif key in {"sex", "gender"}:
            metadata["sex"] = value.lower() if value else "unknown"
        elif key == "dx":
            metadata["dx_codes"] = [code.strip() for code in value.split(",") if code.strip()]
    return metadata


def rhythm_target(dx_codes: Iterable[str]) -> str | None:
    """Return the unique primary rhythm, or None for an ambiguous header.

    In this database the first ``#Dx`` code is the rhythm and subsequent codes
    are additional whole-record conditions.  We also require that no second
    rhythm code occurs later in the list.  This avoids interpreting the shared
    SNOMED code 195101003 as SAAWR when it is used for the non-rhythm WAVN label.
    """
    codes = list(dx_codes)
    if not codes or codes[0] not in SNOMED_TO_CLASS:
        return None
    labels = {SNOMED_TO_CLASS[code] for code in codes if code in SNOMED_TO_CLASS}
    return SNOMED_TO_CLASS[codes[0]] if len(labels) == 1 else None


def resize_signal(signal: np.ndarray, target_length: int) -> np.ndarray:
    """Resample a complete ECG to target_length while retaining all 10 seconds."""
    signal = np.asarray(signal, dtype=np.float32)
    if signal.ndim != 2:
        raise ValueError(f"Expected [time, leads], got {signal.shape}.")
    if signal.shape[0] == target_length:
        return signal
    # Polyphase resampling is deterministic and avoids loading an FFT-sized copy.
    from math import gcd
    divisor = gcd(int(signal.shape[0]), int(target_length))
    resized = resample_poly(signal, target_length // divisor, signal.shape[0] // divisor, axis=0)
    if resized.shape[0] > target_length:
        resized = resized[:target_length]
    elif resized.shape[0] < target_length:
        resized = np.pad(resized, ((0, target_length - resized.shape[0]), (0, 0)))
    return np.asarray(resized, dtype=np.float32)


def load_wfdb_record(record_base: str | Path) -> np.ndarray:
    """Load calibrated physical signals and reorder them to the canonical leads."""
    try:
        import wfdb
    except ImportError as exc:
        raise ImportError("Install the dataset dependency with: pip install wfdb") from exc

    signal, fields = wfdb.rdsamp(str(Path(record_base)), return_res=32)
    signal = np.asarray(signal, dtype=np.float32)
    lead_names = list(fields.get("sig_name", []))
    if signal.shape[1] != 12:
        raise ValueError(f"Expected 12 leads for {record_base}, got {signal.shape[1]}.")
    if lead_names:
        missing = [lead for lead in CANONICAL_LEADS if lead not in lead_names]
        if missing:
            raise ValueError(f"Missing canonical leads {missing} in {record_base}; found {lead_names}.")
        signal = signal[:, [lead_names.index(lead) for lead in CANONICAL_LEADS]]
    return np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)


class ECGArrhythmiaClassificationDataset(BaseDataset):
    """Lazy, record-level classification dataset for 12-lead, 10-second ECGs."""

    supported_tasks = ["classification"]
    description = (
        "The PhysioNet ECG Arrhythmia Database contains 10-second, 12-lead ECG "
        "recordings sampled at 500 Hz. This configuration predicts one of eleven "
        "expert-annotated cardiac rhythm categories."
    )
    task_description = (
        "Classify the complete 12-lead ECG recording into its single annotated "
        "rhythm category."
    )

    def load_data(self) -> None:
        root = Path(_cfg_get(self.dataset_config, "root", "data/ecg-arrhythmia-1.0.0")).expanduser()
        processed_dir = str(_cfg_get(self.dataset_config, "processed_dir", "processed/rhythm_single"))
        self.root = root.resolve()
        self.processed_root = self.root / processed_dir
        manifest_path = self.processed_root / "splits" / f"{self.split}.csv"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Missing {manifest_path}. Run scripts/prepare_ecg_arrhythmia.py --root {self.root} first."
            )
        with manifest_path.open(newline="", encoding="utf-8") as handle:
            self.rows = list(csv.DictReader(handle))
        if not self.rows:
            raise RuntimeError(f"No records in split manifest: {manifest_path}")
        self.labels = torch.tensor([int(row["label_index"]) for row in self.rows], dtype=torch.long)
        self.record_descriptions = [
            "Patient information: " + json.dumps(
                {
                    "age": None if row.get("age", "") in {"", "None", "unknown"} else int(row["age"]),
                    "sex": row.get("sex", "unknown").lower(),
                },
                separators=(",", ":"),
            )
            for row in self.rows
        ]
        self.moment_windows = int(_cfg_get(self.dataset_config, "moment_windows", 0))
        self.normalization_mean = np.zeros(12, dtype=np.float32)
        self.normalization_std = np.ones(12, dtype=np.float32)
        if bool(self.config.data.normalize):
            stats_path = self.processed_root / "normalization.npz"
            if not stats_path.exists():
                raise FileNotFoundError(
                    f"Missing {stats_path}. Re-run preparation without --skip-stats."
                )
            stats = np.load(stats_path)
            self.normalization_mean = np.asarray(stats["mean"], dtype=np.float32)
            self.normalization_std = np.maximum(np.asarray(stats["std"], dtype=np.float32), 1e-6)
            if self.normalization_mean.shape != (12,) or self.normalization_std.shape != (12,):
                raise ValueError(f"Invalid normalization arrays in {stats_path}.")

    def _normalise(self, signal: np.ndarray) -> np.ndarray:
        return ((signal - self.normalization_mean) / self.normalization_std).astype(np.float32)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        record_base = self.root / row["record"]
        raw = load_wfdb_record(record_base)
        main = self._normalise(resize_signal(raw, int(self.history_len)))
        out: dict[str, Any] = {
            "x_enc": torch.from_numpy(main),
            "labels": self.labels[index],
            "descriptions": self.record_descriptions[index],
        }
        if self.moment_windows > 0:
            bounds = np.linspace(0, raw.shape[0], self.moment_windows + 1, dtype=int)
            windows = [
                self._normalise(resize_signal(raw[bounds[i] : bounds[i + 1]], int(self.history_len)))
                for i in range(self.moment_windows)
            ]
            out["x_moment_windows"] = torch.from_numpy(np.stack(windows, axis=0))
        return out

    def __len__(self) -> int:
        return len(self.rows)

    def inverse_index(self, index: int) -> int:
        return index

    @property
    def n_points(self) -> int:
        return len(self.rows)

    @property
    def n_features(self) -> int:
        return 12

    @property
    def n_classes(self) -> int:
        return len(CLASS_CODES)

    @property
    def class_codes(self) -> list[str]:
        return list(CLASS_CODES)

    @property
    def class_names(self) -> list[str]:
        return list(CLASS_NAMES)


ecg_arrhythmia_datasets = {"classification": ECGArrhythmiaClassificationDataset}
