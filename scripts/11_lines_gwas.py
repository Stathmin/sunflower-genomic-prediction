"""Canonical additive GWAS on paternal lines.

The additive scan is GEMMA's LMM score test (``-lmm 4``), computed with
VanRaden-style centred kinship and the first three principal components.  The
no-PC model is reported for comparison.
"""
from __future__ import annotations

import argparse
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from sunflower.external import prepare_panel, run_gemma
from sunflower.genotypes import (
    filter_panel,
    load_genotypes,
    standardized_pca,
    vanraden_kinship,
)
from sunflower.gwas import scan_summary
from sunflower.io import line_trait_means, write_csv

CONF = {"LI29", "LI30"}
TRAITS = [
    "oil_content", "seed_weight_1000", "hull_content", "head_diameter",
    "plant_height", "days_emergence_flowering", "days_central_to_side",
    "autofertility_self", "autofertility_open",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=max(1, os.cpu_count() or 1))
    args = parser.parse_args()

    data = load_genotypes()
    lines54 = data.lines
    lines52 = [line for line in lines54 if line not in CONF]
    means = line_trait_means()

    tasks = []
    with tempfile.TemporaryDirectory(prefix="sunflower-gwas-") as tmp:
        for tag, lines in (("54_with_confectionery", lines54),
                           ("52_oilseed_only", lines52)):
            panel = filter_panel(data, lines, call_rate=0.9, maf=0.05)
            scores, _ = standardized_pca(panel.dosage, 3)
            prepared = prepare_panel(
                panel, tmp, prefix=f"gwas_{tag}",
                kinship=vanraden_kinship(panel.dosage),
            )
            values = {trait: means[trait].reindex(lines).to_numpy(float)
                      for trait in TRAITS}
            for model, covariates in (("3_PC_canon", scores[:, :3]), ("no_PC", None)):
                for trait in TRAITS:
                    tasks.append((tag, model, trait, covariates, prepared, values[trait]))

        def scan(task):
            tag, model, trait, covariates, prepared, y = task
            result = run_gemma(
                prepared, y, covariates=covariates,
                label=f"{tag}_{model}_{trait}",
            )
            summary = scan_summary(result["assoc"], prepared.n_markers)
            summary["h2_SNP"] = result["pve"]
            summary["panel"] = tag
            summary["model"] = model
            summary["trait"] = trait
            return summary

        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            rows = list(pool.map(scan, tasks))

    table = pd.DataFrame(rows).rename(columns={
        "lambda": "λ", "n_p<5e-8": "n_p<5e-8", "n_bonf": "n_bonf",
    })
    table = table[[
        "panel", "model", "trait", "n_SNP", "λ", "h2_SNP",
        "n_p<5e-8", "top", "top_p",
    ]]
    write_csv(table, "lines_gwas.csv")

    for (tag, model), group in table.groupby(["panel", "model"], sort=False):
        print(
            f"{tag:22s} {model:12s} lambda={group['λ'].min():.2f}-{group['λ'].max():.2f}; "
            f"significant at 5e-8: {int(group['n_p<5e-8'].sum())}"
        )
    print("\nwritten: lines_gwas.csv")


if __name__ == "__main__":
    main()
