#!/usr/bin/env python3
"""Report the imports required by the combined ECG experiment."""
from importlib import import_module

REQUIRED = ["torch", "transformers", "peft", "momentfm", "wfdb", "scipy", "sklearn", "toml"]
failed = []
for name in REQUIRED:
    try:
        module = import_module(name)
        print(f"OK {name}: {getattr(module, '__version__', 'version unavailable')}")
    except Exception as exc:
        failed.append((name, repr(exc)))
        print(f"FAIL {name}: {exc}")
if failed:
    raise SystemExit("Missing or broken dependencies: " + ", ".join(name for name, _ in failed))
