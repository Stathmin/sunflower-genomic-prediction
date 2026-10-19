"""Cross-panel dominance scan on the call set without missing genotypes.

The separate VCF is read with pysam; sample names are mapped to the deposit
names.  The scan itself is the same two-kernel model as script 14.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from sunflower.dominance import run_vcf
from sunflower.io import read_csv, write_csv
from sunflower.paths import GENOTYPES

TRAIT = "seed_weight_1000"
WINDOWS = [
    ("chr14:169.20-169.23", 14, 169_200_000, 169_230_000),
    ("chr17:50.20-50.50", 17, 50_200_000, 50_500_000),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nrep", type=int, default=400)
    parser.add_argument("--jobs", type=int, default=max(1, os.cpu_count() or 1))
    args = parser.parse_args()

    blup = read_csv("hybrid_blup_phenotypes.csv")
    result = run_vcf(
        GENOTYPES / "filt_geno_no_imputation.vcf", blup, TRAIT,
        nrep=args.nrep, modes=("bootstrap",), jobs=args.jobs,
        out_tag="_no_imputation",
    )
    write_csv(
        pd.DataFrame({"min_p": result["nulls"]["bootstrap"]}),
        "dominance_null_no_imputation.csv",
    )

    scan = result["scan"]
    null = result["nulls"]["bootstrap"]
    threshold = float(np.quantile(null, 0.05))
    print(
        f"\nEMPIRICAL THRESHOLD on the phase-6 panel ({args.nrep} replicates): "
        f"5%={threshold:.3e}  1%={np.quantile(null, 0.01):.3e}"
    )
    for name, chromosome, low, high in WINDOWS:
        window = scan[
            (scan["chr"] == chromosome) & (scan["pos"].between(low, high))
        ].dropna(subset=["p_dom"])
        if window.empty:
            print(f"  {name}: no testable SNP")
            continue
        best = window.loc[window["p_dom"].idxmin()]
        empirical = (np.sum(null <= best["p_dom"]) + 1) / (args.nrep + 1)
        print(
            f"  {name:18s} p_dom={best['p_dom']:.3e}  empirical genome-wide "
            f"p={empirical:.4f}  -> "
            f"{'SIGNIFICANT' if empirical < 0.05 else 'NOT SIGNIFICANT'}"
        )


if __name__ == "__main__":
    main()
