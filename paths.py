"""Compatibility shim: import :mod:`sunflower.paths` instead.

The package is installed with ``uv sync``; there is no import-time directory
walking or ``sys.path`` manipulation.
"""
from sunflower.paths import *  # noqa: F401,F403
