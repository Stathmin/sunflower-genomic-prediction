"""Paths used by the scripts. Run every script from any directory; paths resolve relative to this file.

tables/     phenotype tables (inputs) and tables written by the scripts (outputs); see README.md
genotypes/  genotype call sets (VCF, NPZ)                                    [genomic repository only]
Undeposited primary files (field workbooks, the two separate call sets before merging) are needed only by
scripts/provenance/. Point SUNFLOWER_WORK in a local .env file (see .env.example) at the folder that holds them.
"""
from __future__ import annotations
import os, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load_env() -> None:
    env = HERE / ".env"
    if env.exists():
        for raw in env.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_env()
OUT = INTERMEDIATE = HERE / "tables"
OUT.mkdir(exist_ok=True)
DEPOSIT = HERE / "genotypes" if (HERE / "genotypes").exists() else OUT
ROOT = DATA = D = DEPOSIT
_w = os.environ.get("SUNFLOWER_WORK", "").strip()
WORK = Path(_w).expanduser() if _w else None
_c = os.environ.get("SUNFLOWER_CALLING", "").strip()
CALLING = Path(_c).expanduser() if _c else None
MODELS = HERE / "scripts" / "models"

from names import en_table, en_ids, en_id, en_trait, ru_id, ru_col, ru_trait, is_line_id  # noqa: E402,F401


def use_models() -> None:
    if str(MODELS) not in sys.path:
        sys.path.insert(0, str(MODELS))


def work() -> Path:
    if WORK is None:
        raise RuntimeError("This script reads an undeposited primary file: set SUNFLOWER_WORK in .env.")
    return WORK


def calling() -> Path:
    if CALLING is None:
        raise RuntimeError("Set SUNFLOWER_CALLING in .env for the variant-calling tree.")
    return CALLING
