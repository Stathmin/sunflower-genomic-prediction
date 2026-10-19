"""Data locations for the pipeline.

Run from the repository root, or set ``SUNFLOWER_ROOT`` to the directory that
contains ``tables/`` and ``genotypes/``.  The previous per-script search for a
``paths.py`` file was removed; the package is installed with ``uv`` instead.
"""
from __future__ import annotations

import os
from pathlib import Path

from .names import (  # noqa: F401  (re-exported for scripts)
    en_id,
    en_ids,
    en_table,
    en_trait,
    is_line_id,
    ru_col,
    ru_id,
    ru_trait,
)


def _root() -> Path:
    env = os.environ.get("SUNFLOWER_ROOT", "").strip()
    return Path(env).expanduser().resolve() if env else Path.cwd().resolve()


ROOT = _root()
TABLES = ROOT / "tables"
GENOTYPES = ROOT / "genotypes"
FIGURES = ROOT / "figures"
LOGS = ROOT / "logs"

WORK = Path(os.environ["SUNFLOWER_WORK"]).expanduser().resolve() if os.environ.get("SUNFLOWER_WORK") else None
CALLING = Path(os.environ["SUNFLOWER_CALLING"]).expanduser().resolve() if os.environ.get("SUNFLOWER_CALLING") else None


def require_root() -> Path:
    """Fail early with an actionable message instead of guessing a directory."""
    missing = [str(p) for p in (TABLES, GENOTYPES) if not p.is_dir()]
    if missing:
        raise RuntimeError(
            "Sunflower data not found: "
            + ", ".join(missing)
            + ". Run from the repository root or set SUNFLOWER_ROOT."
        )
    return ROOT


def require_work() -> Path:
    if WORK is None:
        raise RuntimeError(
            "This step reads undeposited primary files: set SUNFLOWER_WORK in .env."
        )
    return WORK


def require_calling() -> Path:
    if CALLING is None:
        raise RuntimeError(
            "This step reads the variant-calling tree: set SUNFLOWER_CALLING in .env."
        )
    return CALLING


def table(name: str) -> Path:
    return TABLES / name


def genotype(name: str) -> Path:
    return GENOTYPES / name
