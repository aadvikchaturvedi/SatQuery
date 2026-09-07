"""
Side-effect-only import: makes the existing, untouched
`vqa_and_change_using_gemini` code importable from the backend without
moving or copying any of it. Import this module before importing anything
from `vqa_and_change_using_gemini`.

Two paths are needed because that package is internally inconsistent about
how its own modules import each other (pre-existing, not something this
backend changes): `pipeline.py` uses the qualified
`from vqa_and_change_using_gemini.imaging import ...`, while `vqa.py` uses
the bare `from imaging import ...`. The former needs the repo root on
sys.path; the latter needs the package directory itself on sys.path.
"""
import sys

from app.config import REPO_ROOT

_ML_PACKAGE_DIR = REPO_ROOT / "vqa_and_change_using_gemini"

for _path in (str(REPO_ROOT), str(_ML_PACKAGE_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)
