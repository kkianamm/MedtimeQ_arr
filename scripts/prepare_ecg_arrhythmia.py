#!/usr/bin/env python3
"""Build deterministic single-rhythm splits and train-set normalization stats."""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# Run from the repository root after the installer copies this script.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.ecg_arrhythmia import (  # noqa: E402
    CLASS_CODES,
    CLASS_TO_INDEX,
    load_wfdb_record,
    parse_header,
    resize_signal,
    rhythm_target,
)

FIELDS = ["record", "label", "label_index", "age", "sex", "dx_codes", "sampling_rate", "signal_length"]


def allocate_splits(rows, seed: int, val_fraction: float, test_fraction: float):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["label"]].append(row)
    rng = random.Random(seed)
    result = {"train": [], "val": [], "test": []}
    for label in CLASS_CODES:
        class_rows = grouped[label]
        rng.shuffle(class_rows)
        n = len(class_rows)
        if n < 3:
            raise RuntimeError(f"Class {label} has only {n} records; at least 3 are required.")
        n_test = max(1, int(round(n * test_fraction)))
        n_val = max(1, int(round(n * val_fraction)))
        while n_test + n_val >= n:
            if n_test >= n_val and n_test > 1:
                n_test -= 1
            elif n_val > 1:
                n_val -= 1
            else:
                raise RuntimeError(f"Cannot split class {label} with n={n}.")
        result["test"].extend(class_rows[:n_test])
        result["val"].extend(class_rows[n_test : n_test + n_val])
        result["train"].extend(class_rows[n_test + n_val :])
    for split_rows in result.values():
        rng.shuffle(split_rows)
    return result


def write_manifest(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def compute_stats(root: Path, train_rows, history_len: int):
    total = 0
    channel_sum = np.zeros(12, dtype=np.float64)
    channel_sumsq = np.zeros(12, dtype=np.float64)
    for number, row in enumerate(train_rows, 1):
        signal = resize_signal(load_wfdb_record(root / row["record"]), history_len).astype(np.float64)
        total += signal.shape[0]
        channel_sum += signal.sum(axis=0)
        channel_sumsq += np.square(signal).sum(axis=0)
        if number % 1000 == 0:
            print(f"normalization: {number}/{len(train_rows)} records")
    mean = channel_sum / total
    variance = np.maximum(channel_sumsq / total - np.square(mean), 1e-12)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32), total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="Dataset directory containing WFDBRecords")
    parser.add_argument("--processed-dir", default="processed/rhythm_single")
    parser.add_argument("--history-len", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--test-fraction", type=float, default=0.10)
    parser.add_argument("--skip-stats", action="store_true", help="Metadata/split smoke test only")
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    records_root = root / "WFDBRecords" if (root / "WFDBRecords").is_dir() else root
    headers = sorted(records_root.rglob("*.hea"))
    if not headers:
        raise SystemExit(f"No .hea files found below {records_root}")

    rows, exclusions = [], Counter()
    for header in headers:
        meta = parse_header(header)
        if meta["n_signals"] != 12:
            exclusions["not_12_lead"] += 1
            continue
        label = rhythm_target(meta["dx_codes"])
        if label is None:
            exclusions["zero_or_multiple_rhythm_labels"] += 1
            continue
        base = header.relative_to(root).with_suffix("").as_posix()
        rows.append(
            {
                "record": base,
                "label": label,
                "label_index": CLASS_TO_INDEX[label],
                "age": "" if meta["age"] is None else meta["age"],
                "sex": meta["sex"],
                "dx_codes": ",".join(meta["dx_codes"]),
                "sampling_rate": meta["sampling_rate"],
                "signal_length": meta["signal_length"],
            }
        )

    counts = Counter(row["label"] for row in rows)
    missing = [code for code in CLASS_CODES if counts[code] < 3]
    if missing:
        raise SystemExit(f"Insufficient records for classes: {[(code, counts[code]) for code in missing]}")
    splits = allocate_splits(rows, args.seed, args.val_fraction, args.test_fraction)
    processed = root / args.processed_dir
    for split, split_rows in splits.items():
        write_manifest(processed / "splits" / f"{split}.csv", split_rows)

    summary = {
        "root": str(root),
        "label_mode": "single_unique_rhythm",
        "seed": args.seed,
        "history_len": args.history_len,
        "class_order": CLASS_CODES,
        "eligible_records": len(rows),
        "excluded": dict(exclusions),
        "all_counts": dict(counts),
        "split_counts": {
            split: dict(Counter(row["label"] for row in split_rows))
            for split, split_rows in splits.items()
        },
    }
    processed.mkdir(parents=True, exist_ok=True)
    (processed / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if not args.skip_stats:
        mean, std, samples = compute_stats(root, splits["train"], args.history_len)
        np.savez(processed / "normalization.npz", mean=mean, std=std, samples=np.asarray(samples))
        summary["normalization_mean"] = mean.tolist()
        summary["normalization_std"] = std.tolist()
        (processed / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    if args.skip_stats:
        print("NOTE: --skip-stats was used; training with data.normalize=true will require a full preparation run.")


if __name__ == "__main__":
    main()
