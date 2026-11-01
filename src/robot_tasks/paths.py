"""Repository resources for the editable source checkout.

Configs and manufacturer assets stay outside src/. Install with pip install -e
so all entry points resolve the same checkout, independent of the working dir.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
