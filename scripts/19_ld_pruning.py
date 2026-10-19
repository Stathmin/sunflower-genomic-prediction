"""Sensitivity of paternal GCA prediction to LD pruning.

Pruning is delegated to PLINK 1.9 ``--indep-pairwise`` (windows of 50 and 1000
SNPs, r2 > 0.5 and 0.2).  The GBLUP and VanRaden kinship are the shared
functions in ``sunflower.prediction``.
"""
from __future__ import annotations

import argparse
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from sunflower.external import plink_prune, prepare_panel
from sunflower.genotypes import (
    filter_panel,
    load_genotypes,
    standardized_pca,
    subset_panel,
    vanraden_kinship,
)
from sunflower.io import read_csv, write_csv
from sunflower.prediction import CONF, gblup_cv

TRAITS = ["seed_yield", "oil_content", "seed_weight_1000", "oil_yield"]
CONFIGS = [
    (None, None, None),
    (50, 5, 0.5),
    (50, 5, 0.2),
    (1000, 50, 0.5),
    (1000, 50, 0.2),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=max(1, os.cpu_count() or 1))
    args = parser.parse_args()

    data = load_genotypes()
    blup = read_csv("hybrid_blup_phenotypes.csv")
    blup[["mother_long", "father"]] = blup["hybrid"].str.split("_", n=1, expand=True)
    fathers = sorted(blup["father"].unique())
    kept_lines = [line for line in fathers if line not in CONF]

    # The filter is the 57-parent panel of script 06; the 54 paternal columns
    # are then selected, matching the deposited 82,889-SNP set.
    panel = subset_panel(filter_panel(data, call_rate=0.9, maf=0.05), fathers)
    print(f"panel: {panel.n_markers:,} SNP x {len(fathers)} paternal lines")

    rows = []
    with tempfile.TemporaryDirectory(prefix="sunflower-ld-") as tmp:
        prepared = prepare_panel(
            panel, tmp, prefix="ld",
            kinship=vanraden_kinship(panel.dosage),
        )
        for window, step, threshold in CONFIGS:
            if window is None:
                keep = np.ones(panel.n_markers, dtype=bool)
            else:
                keep = plink_prune(
                    prepared, window=window, step=step, r2=threshold,
                    label=f"prune_{window}_{step}_{threshold}",
                )
            dosage = panel.dosage[keep]
            kinship = vanraden_kinship(dosage) + np.eye(len(fathers)) * 1e-6
            _, variance = standardized_pca(dosage, 2)

            def one(trait: str, kinship=kinship):
                means = blup.dropna(subset=[trait]).groupby("father")[trait].mean()
                lines = [line for line in kept_lines if line in means.index]
                y = means.reindex(lines).to_numpy(float)
                index = [fathers.index(line) for line in lines]
                k = kinship[np.ix_(index, index)]
                predicted = np.array([
                    gblup_cv(y, k, [j for j in range(len(y)) if j != i], [i])[0]
                    for i in range(len(y))
                ])
                return float(np.corrcoef(predicted, y)[0, 1])

            with ThreadPoolExecutor(max_workers=args.jobs) as pool:
                correlations = dict(zip(TRAITS, pool.map(one, TRAITS)))
            row = {
                "window_snps": window, "step": step, "r2_threshold": threshold,
                "snps": int(keep.sum()),
                "pc1_pct": float(variance[0] * 100),
                "pc2_pct": float(variance[1] * 100),
                **{f"r_{trait}": value for trait, value in correlations.items()},
            }
            rows.append(row)
            print("  " + ", ".join(
                f"{key}={value:.3f}" if isinstance(value, float) else f"{key}={value}"
                for key, value in row.items()
            ), flush=True)

    table = pd.DataFrame(rows)
    expected = [0.472, 0.404, 0.655, 0.480]
    observed = [float(table.loc[0, f"r_{trait}"]) for trait in TRAITS]
    assert np.allclose(observed, expected, atol=3e-3), (observed, expected)
    write_csv(table, "ld_pruning_sensitivity.csv")
    print("\nsaved tables/ld_pruning_sensitivity.csv")


if __name__ == "__main__":
    main()
