"""Vercel serverless entry point.

Vercel's Python runtime serves a WSGI `app` object exported from this module.
We simply expose the existing Flask app — no backend logic lives here.

The project root is added to sys.path so `import src...` resolves when Vercel
runs this file from the api/ directory.
"""
import os
import sys

# Ensure the project root (parent of api/) is importable.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.web.server import app  # noqa: E402  (must follow sys.path setup)

# Vercel looks for a module-level `app` (WSGI callable).
__all__ = ["app"]
