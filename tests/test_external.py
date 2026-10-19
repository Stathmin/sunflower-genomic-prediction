import shutil

import numpy as np
import pytest

from sunflower.external import plink_prune, prepare_panel, run_gemma
from sunflower.genotypes import Panel, vanraden_kinship

pytestmark = pytest.mark.external


def _tools_available() -> bool:
    return bool(
        shutil.which("gemma")
        and (shutil.which("plink1.9") or shutil.which("plink"))
    )


def _panel(n_samples: int = 20, n_markers: int = 60) -> Panel:
    rng = np.random.default_rng(6)
    dosage = (rng.random((n_markers, n_samples)) > 0.5).astype(float)
    return Panel(
        dosage=dosage, raw=dosage,
        chrom=np.ones(n_markers, dtype=int),
        pos=np.arange(1, n_markers + 1),
        samples=tuple(f"s{i}" for i in range(n_samples)),
        mask=np.ones(n_markers, dtype=bool),
        missing_fraction=0.0,
    )


def test_gemma_and_plink_round_trip(tmp_path):
    if not _tools_available():
        pytest.skip("gemma and plink1.9 are required")
    panel = _panel()
    prepared = prepare_panel(
        panel, tmp_path, prefix="tiny",
        kinship=vanraden_kinship(panel.dosage),
    )
    rng = np.random.default_rng(7)
    phenotype = panel.dosage[0] + rng.normal(size=len(panel.samples))
    result = run_gemma(prepared, phenotype, label="scan")
    assoc = result["assoc"]
    assert len(assoc) == panel.n_markers
    assert np.isfinite(assoc["p"]).any()

    keep = plink_prune(prepared, window=10, step=5, r2=0.9, label="prune")
    assert keep.shape == (panel.n_markers,)
    assert keep.sum() > 0
