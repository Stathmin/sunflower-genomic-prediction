import numpy as np
import pandas as pd

from sunflower.dominance import build_kernels, fit_vc, scan
from sunflower.genotypes import Panel


def _panel(n_samples=30, n_markers=100):
    rng = np.random.default_rng(4)
    dosage = (rng.random((n_markers, n_samples)) > 0.5).astype(float)
    return Panel(
        dosage=dosage, raw=dosage,
        chrom=np.ones(n_markers, dtype=int),
        pos=np.arange(n_markers),
        samples=tuple(f"s{i}" for i in range(n_samples)),
        mask=np.ones(n_markers, dtype=bool),
        missing_fraction=0.0,
    )


def test_two_kernel_scan_runs():
    panel = _panel()
    n_hybrids = 30
    frame = pd.DataFrame({
        "mother": [f"s{i % 3}" for i in range(n_hybrids)],
        "father": [f"s{3 + i % 5}" for i in range(n_hybrids)],
        "trait": np.random.default_rng(5).normal(size=n_hybrids),
    })
    kernels = build_kernels(panel, frame, "trait")
    assert kernels.ka.shape == (n_hybrids, n_hybrids)
    assert kernels.kd.shape == (n_hybrids, n_hybrids)
    ha, hd = fit_vc(kernels.y, kernels, starts=((0.5, 0.2), (0.2, 0.5)))
    assert 0 <= ha < 1 and 0 <= hd < 1
    pvalues, betas = scan(kernels.y, ha, hd, kernels)
    assert pvalues.shape == (panel.n_markers,)
    assert betas.shape == (panel.n_markers,)
