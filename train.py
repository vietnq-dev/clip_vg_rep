"""Compatibility shim so ``python train.py`` matches the console script."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

from visual_grounding.__main__ import main


if __name__ == "__main__":
    main()
