#!/usr/bin/env python3
"""Install the ECG-arrhythmia adaptation into medtsllm-biomedcoop."""
from __future__ import annotations

import argparse
import ast
import re
import shutil
from pathlib import Path


def backup(path: Path) -> None:
    backup_path = path.with_name(path.name + ".before_ecg_arrhythmia.bak")
    if path.exists() and not backup_path.exists():
        shutil.copy2(path, backup_path)


def copy_file(source: Path, destination: Path, *, backup_existing: bool = False) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if backup_existing and destination.exists():
        backup(destination)
    shutil.copy2(source, destination)


def patch_registry(path: Path, import_line: str, mapping_key: str, mapping_value: str) -> None:
    backup(path)
    text = path.read_text(encoding="utf-8")
    if import_line not in text:
        lines = text.splitlines(keepends=True)
        insert_at = 0
        for index, line in enumerate(lines):
            if line.startswith("from .") or line.startswith("import "):
                insert_at = index + 1
        lines.insert(insert_at, import_line)
        text = "".join(lines)
    key_single = f"'{mapping_key}'"
    key_double = f'"{mapping_key}"'
    if key_single not in text and key_double not in text:
        marker = "model_lookup = {" if mapping_value == "TriMedTsLLM" else "dataset_lookup = {"
        marker_index = text.find(marker)
        if marker_index < 0:
            raise RuntimeError(f"Could not find {marker.split()[0]} in {path}")
        brace_index = text.find("{", marker_index)
        insertion = f'\n    "{mapping_key}": {mapping_value},'
        text = text[: brace_index + 1] + insertion + text[brace_index + 1 :]
    ast.parse(text)
    path.write_text(text, encoding="utf-8")


def verify_host(repo: Path) -> None:
    required = [
        repo / "models" / "medtsllm.py",
        repo / "models" / "biomedcoop_ts.py",
        repo / "models" / "__init__.py",
        repo / "datasets" / "base.py",
        repo / "datasets" / "__init__.py",
        repo / "tasks" / "classification.py",
        repo / "train.py",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("Not a compatible medtsllm-biomedcoop checkout; missing: " + ", ".join(missing))
    checks = {
        repo / "models" / "medtsllm.py": ["BiomedCoOpHead", "_build_class_prototypes"],
        repo / "models" / "biomedcoop_ts.py": ["class BiomedCoOpHead", "statistics_based_prompt_selection"],
        repo / "tasks" / "classification.py": ["f1_score", "self.history"],
    }
    for path, markers in checks.items():
        text = path.read_text(encoding="utf-8")
        absent = [marker for marker in markers if marker not in text]
        if absent:
            raise SystemExit(f"{path} is not the expected reference implementation; missing markers: {absent}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo", type=Path, help="Path to a medtsllm-biomedcoop checkout")
    parser.add_argument(
        "--data-root",
        default="data/ecg-arrhythmia-1.0.0",
        help="Dataset root copied into the generated TOML (absolute paths are accepted)",
    )
    parser.add_argument(
        "--sync-classification-task",
        action="store_true",
        help="Replace tasks/classification.py with the checked reference copy after making a backup",
    )
    args = parser.parse_args()

    repo = args.repo.expanduser().resolve()
    here = Path(__file__).resolve().parent
    verify_host(repo)

    copy_file(here / "datasets" / "ecg_arrhythmia.py", repo / "datasets" / "ecg_arrhythmia.py")
    copy_file(here / "models" / "tri_components.py", repo / "models" / "tri_components.py", backup_existing=True)
    copy_file(here / "models" / "tri_medtsllm.py", repo / "models" / "tri_medtsllm.py", backup_existing=True)
    copy_file(here / "prompts" / "ecg_arrhythmia_rhythm.json", repo / "prompts" / "ecg_arrhythmia_rhythm.json")
    copy_file(here / "scripts" / "prepare_ecg_arrhythmia.py", repo / "scripts" / "prepare_ecg_arrhythmia.py")
    copy_file(here / "scripts" / "check_ecg_arrhythmia.py", repo / "scripts" / "check_ecg_arrhythmia.py")
    copy_file(here / "scripts" / "check_environment.py", repo / "scripts" / "check_environment.py")
    copy_file(here / "scripts" / "train_ecg_arrhythmia.sh", repo / "scripts" / "train_ecg_arrhythmia.sh")
    copy_file(here / "requirements-ecg-arrhythmia.txt", repo / "requirements-ecg-arrhythmia.txt")
    copy_file(here / "requirements-combined-model.txt", repo / "requirements-combined-model.txt")
    if args.sync_classification_task:
        copy_file(here / "tasks" / "classification.py", repo / "tasks" / "classification.py", backup_existing=True)

    config_source = here / "configs" / "datasets" / "ecg_arrhythmia_combined.toml"
    config_destination = repo / "configs" / "datasets" / "ecg_arrhythmia_combined.toml"
    config_destination.parent.mkdir(parents=True, exist_ok=True)
    config_text = config_source.read_text(encoding="utf-8")
    escaped_root = str(Path(args.data_root).expanduser())
    config_text = re.sub(
        r'(?m)^root\s*=\s*"[^"]*"',
        'root = ' + repr(escaped_root).replace("'", '"'),
        config_text,
        count=1,
    )
    config_destination.write_text(config_text, encoding="utf-8")

    patch_registry(
        repo / "models" / "__init__.py",
        "from .tri_medtsllm import TriMedTsLLM\n",
        "tri_medtsllm",
        "TriMedTsLLM",
    )
    patch_registry(
        repo / "datasets" / "__init__.py",
        "from .ecg_arrhythmia import ecg_arrhythmia_datasets\n",
        "ECG-Arrhythmia",
        "ecg_arrhythmia_datasets",
    )

    print(f"Installed ECG-arrhythmia adaptation into: {repo}")
    print(f"Configuration: {config_destination}")
    print(f"Dataset root: {escaped_root}")
    print("Next command:")
    print(f"  python3 scripts/prepare_ecg_arrhythmia.py --root {escaped_root}")


if __name__ == "__main__":
    main()
