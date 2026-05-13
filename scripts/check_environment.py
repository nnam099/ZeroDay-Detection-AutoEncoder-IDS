from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


def module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def collect_environment() -> dict:
    packages = ["torch", "numpy", "pandas", "sklearn", "matplotlib", "streamlit", "shap", "dotenv"]
    provider_keys = {
        "groq": "GROQ_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    key_name = provider_keys.get(provider)

    return {
        "python": sys.version.split()[0],
        "root": str(ROOT_DIR),
        "packages": {name: module_available(name) for name in packages},
        "artifacts": {
            "v14_model": (ROOT_DIR / "checkpoints" / "ids_v14_model.pth").exists(),
            "v14_pipeline": (ROOT_DIR / "checkpoints" / "ids_v14_pipeline.pkl").exists(),
            "v15_model": (ROOT_DIR / "checkpoints" / "ids_v15_model.pth").exists(),
            "v15_pipeline": (ROOT_DIR / "checkpoints" / "ids_v15_pipeline.pkl").exists(),
        },
        "data": {
            "data_dir": (ROOT_DIR / "data").exists(),
            "training_csv": (ROOT_DIR / "data" / "UNSW_NB15_training-set.csv").exists(),
            "testing_csv": (ROOT_DIR / "data" / "UNSW_NB15_testing-set.csv").exists(),
        },
        "llm": {
            "provider": provider or None,
            "key_env": key_name,
            "key_present": bool(os.getenv(key_name, "").strip()) if key_name else False,
        },
    }


def main() -> int:
    print(json.dumps(collect_environment(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
