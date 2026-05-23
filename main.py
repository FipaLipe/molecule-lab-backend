"""Compatibility entrypoint for local development.

Run with:

    uvicorn main:app --reload --port 8000
"""

from pathlib import Path
import sys

src_path = Path(__file__).resolve().parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from molecule_lab.api.app import app

__all__ = ["app"]
