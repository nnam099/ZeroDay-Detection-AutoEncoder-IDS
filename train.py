"""Repository-level launcher for IDS v14 training."""

import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

SPEC = importlib.util.spec_from_file_location("ids_train_entry", os.path.join(SRC, "train.py"))
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)

save_artifacts = MODULE.save_artifacts
run_full = MODULE.run_full
run_demo = MODULE.run_demo
main = MODULE.main

if __name__ == "__main__":
    main()
