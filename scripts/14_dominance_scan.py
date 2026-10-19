"""Dominance-deviation scan for one trait, with three null distributions."""
from __future__ import annotations

import argparse
import os

import numpy as np

from sunflower.dominance import MODES, run
from sunflower.genotypes import filter_panel, load_genotypes
from sunflower.io import read_csv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trait", default="seed_weight_1000")
    parser.add_argument("--phenotypes", default="hybrid_blup_phenotypes.csv")
    parser.add_argument("--nrep", type=int, default=None)
    parser.add_argument("--jobs", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--modes", default=None, help="comma-separated null modes")
    args = parser.parse_args()

    modes = tuple(args.modes.split(",")) if args.modes else tuple(
        os.environ.get("DOM_MODES", ",".join(MODES)).split(",")
    )
    nrep = args.nrep if args.nrep is not None else int(os.environ.get("DOM_NREP", "500"))

    data = load_genotypes()
    panel = filter_panel(data, call_rate=0.9, maf=0.05)
    blup = read_csv(args.phenotypes)
    result = run(panel, blup, args.trait, nrep=nrep, modes=modes, jobs=args.jobs)

    scan = result["scan"]
    best = int(result["best"])
    observed = float(scan["p_dom"].iloc[best])
    print(
        f"\nRESULT chr14 (Ka+Kd model, {nrep} replicates, {args.trait})\n"
        f"  observed best p_dom = {observed:.3e} @ "
        f"chr{int(scan['chr'].iloc[best])}:{int(scan['pos'].iloc[best])}"
    )
    for mode, values in result["nulls"].items():
        empirical = (np.sum(values <= observed) + 1) / (nrep + 1)
        print(
            f"  [{mode:11s}] thr5%={np.quantile(values, 0.05):.2e}  "
            f"thr1%={np.quantile(values, 0.01):.2e}  genome-wide p={empirical:.4f}  -> "
            f"{'SIGNIFICANT' if empirical < 0.05 else 'NOT SIGNIFICANT'}"
        )


if __name__ == "__main__":
    main()
