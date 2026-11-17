"""Shared setup for the organiser scripts' tests.

The scripts live in ``scripts/`` (not a package) and are run as files, so that
directory is put on ``sys.path`` to import them.
"""

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
