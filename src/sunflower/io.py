"""Small, shared I/O helpers.

All deposited tables are UTF-8 with a BOM where Excel compatibility matters;
the readers/writers here keep that convention in one place.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .paths import TABLES


def read_csv(name: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(TABLES / name, encoding="utf-8-sig", **kwargs)


def write_csv(df: pd.DataFrame, name: str, **kwargs) -> Path:
    path = TABLES / name
    df.to_csv(path, index=False, encoding="utf-8-sig", **kwargs)
    return path


def read_parquet(name: str, **kwargs) -> pd.DataFrame:
    return pd.read_parquet(TABLES / name, **kwargs)


def write_parquet(df: pd.DataFrame, name: str, **kwargs) -> Path:
    path = TABLES / name
    df.to_parquet(path, index=False, **kwargs)
    return path


def line_trait_means() -> pd.DataFrame:
    """Line phenotypes averaged over years, one column per trait."""
    from .names import en_table

    frame = en_table(read_parquet("line_phenotypes.parquet"))
    return frame.groupby(["genotype", "trait"])["value"].mean().unstack()


def read_json(name: str):
    return json.loads((TABLES / name).read_text(encoding="utf-8"))


def write_json(obj, name: str) -> Path:
    path = TABLES / name
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    return path
