from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print(f"+ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT_DIR, check=True)


def main() -> int:
    try:
        run([sys.executable, "-m", "compileall", "src", "dashboard", "export_model.py", "patch_checkpoint.py", "tests"])
        run([sys.executable, "-m", "unittest", "discover", "-s", "tests"])
    except subprocess.CalledProcessError as exc:
        return int(exc.returncode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
