"""Confectionery artefact: scans with and without LI29/LI30 (no PC covariates).

The scan is GEMMA's LMM score test; the empirical threshold is a parametric
bootstrap under GEMMA's fitted null model.
"""
from __future__ import annotations

import argparse
import os

from sunflower.genotypes import load_genotypes
from sunflower.gwas import confectionery_scan
from sunflower.io import line_trait_means, write_csv

CONF = {"LI29", "LI30"}
TRAITS = [
    "oil_content", "hull_content", "seed_weight_1000", "head_diameter",
    "plant_height", "days_emergence_flowering", "autofertility_self",
    "autofertility_open",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nboot", type=int, default=200)
    parser.add_argument("--jobs", type=int, default=max(1, os.cpu_count() or 1))
    args = parser.parse_args()

    data = load_genotypes()
    means = line_trait_means()
    line_sets = [
        ("54_with_confectionery", data.lines),
        ("52_oilseed_only", [line for line in data.lines if line not in CONF]),
    ]
    table = confectionery_scan(
        data, means, TRAITS, line_sets,
        with_pcs=False, nboot=args.nboot, jobs=args.jobs,
    )
    write_csv(table, "use_type_artefact.csv")
    print("\n=== SUMMARY (confectionery artefact on merged) ===")
    print(table.pivot(index="trait", columns="callset",
                      values=["n_p<5e-8", "n_below_emp"]).to_string())
    print("\nWritten: use_type_artefact.csv")


if __name__ == "__main__":
    main()
