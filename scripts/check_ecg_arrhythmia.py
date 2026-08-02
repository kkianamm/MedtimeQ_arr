#!/usr/bin/env python3
"""Check split manifests, class order, shapes, and one lazy waveform read."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

CLASS_CODES = ["SB", "SR", "AFIB", "ST", "AF", "SA", "SVT", "AT", "AVNRT", "AVRT", "SAAWR"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--processed-dir", default="processed/rhythm_single")
    parser.add_argument("--read-waveform", action="store_true")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    processed = root / args.processed_dir
    summary = json.loads((processed / "summary.json").read_text())
    assert summary["class_order"] == CLASS_CODES, summary["class_order"]
    print("label mode:", summary["label_mode"])
    print("eligible records:", summary["eligible_records"])
    for split in ("train", "val", "test"):
        path = processed / "splits" / f"{split}.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        counts = Counter(row["label"] for row in rows)
        missing = [code for code in CLASS_CODES if counts[code] == 0]
        if missing:
            raise RuntimeError(f"{split} is missing classes: {missing}")
        print(f"{split:>5}: records={len(rows):6d}, counts={dict(counts)}")
    stats = np.load(processed / "normalization.npz")
    print("normalization mean shape:", stats["mean"].shape)
    print("normalization std shape:", stats["std"].shape)
    if args.read_waveform:
        import wfdb
        with (processed / "splits" / "train.csv").open(newline="", encoding="utf-8") as handle:
            row = next(csv.DictReader(handle))
        signal, fields = wfdb.rdsamp(str(root / row["record"]))
        print("sample waveform:", signal.shape, fields.get("fs"), fields.get("sig_name"))


if __name__ == "__main__":
    main()
