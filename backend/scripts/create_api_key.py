#!/usr/bin/env python3
"""Prints a new random API key to put in BACKEND_API_KEY. Run from backend/."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.security import generate_api_key  # noqa: E402

if __name__ == "__main__":
    print(generate_api_key())
