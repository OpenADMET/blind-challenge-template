"""Shared setup for the HF Space tests.

The Space's modules import each other as top-level modules (``from config import
...``), the way they run inside the Space container, so ``hf_space/`` has to be on
``sys.path`` rather than imported as a package.
"""

import sys
from pathlib import Path

HF_SPACE_DIR = Path(__file__).resolve().parents[2] / "hf_space"
if str(HF_SPACE_DIR) not in sys.path:
    sys.path.insert(0, str(HF_SPACE_DIR))
