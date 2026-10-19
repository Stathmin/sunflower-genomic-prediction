"""Association scans of the lines on alternative marker panels.

Additive scans are run by GEMMA (``-lmm 4``, the EMMAX-equivalent score test).
The reported loci are no longer hard-coded: the script writes the top hits of
each scan, and an optional ``--loci`` file can be supplied to annotate known
positions.
"""
from __future__ import annotations

import argparse
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from sunflower.external import prepare_panel, run_gemma
from sunflower.genotypes import (
    filter_panel,
    load_genotypes,
    standardized_pca,
    vanraden_kinship,
)
from sunflower.gwas import loci_table, scan_summary, top_hits
from sunflower.io import line_trait_means, write_csv
from sunflower.paths import TABLES

CONF = {"LI29", "LI30"}
TRAITS = [
    "oil_content", "hull_content", "seed_weight_1000", "plant_height",
    "head_diameter", "days_emergence_flowering", "days_central_to_side",
    "autofertility_self", "autofertility_open",
]
PANELS = [
    ("call rate >= 0.9", 0.9, False),
    ("call rate >= 0.8", 0.8, False),
    ("call rate >= 0.5", 0.5, False),
    ("full_coverage", 1.0, True),
]


def _audit(data, lines: list[str], call_rate: float, full: bool) -> dict:
    columns = [data.column(line) for line in lines]
    raw = data.dosage[:, columns]
    with np.errstate(invalid="ignore", divide="ignore"):
        counts = np.sum(~np.isnan(raw), axis=1)
        call = counts / raw.shape[1]
        af = np.divide(
            np.nansum(raw, axis=1), 2.0 * counts,
            out=np.full(raw.shape[0], np.nan), where=counts > 0,
        )
        minor = np.minimum(af, 1.0 - af)
    on_chromosome = (data.chrom >= 1) & (data.chrom <= 17)
    step1 = on_chromosome & ((call == 1.0) if full else (call >= call_rate))
    step2 = step1 & (minor >= 0.05)
    return {
        "n_start": len(call),
        "drop_chr": int(np.sum(~on_chromosome)),
        "drop_cr": int(np.sum(on_chromosome) - np.sum(step1)),
        "drop_maf": int(np.sum(step1) - np.sum(step2)),
        "missing": float(np.isnan(raw[step2]).mean()) if step2.any() else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--loci", type=str, default=None,
                        help="optional CSV with locus,chr,pos[,window] annotations")
    args = parser.parse_args()

    data = load_genotypes()
    lines54 = data.lines
    lines52 = [line for line in lines54 if line not in CONF]
    means = line_trait_means()
    loci = pd.read_csv(args.loci) if args.loci else None

    rows_panel, rows_scan, rows_hits, rows_loci = [], [], [], []
    with tempfile.TemporaryDirectory(prefix="sunflower-panels-") as tmp:
        for index, (panel_name, call_rate, full) in enumerate(PANELS):
            for line_name, lines in (("54_lines", lines54), ("52_lines", lines52)):
                panel = filter_panel(data, lines, call_rate=call_rate, maf=0.05)
                audit = _audit(data, lines, call_rate, full)
                scores, variance = standardized_pca(panel.dosage, 3)
                conf = np.array([1.0 if line in CONF else 0.0 for line in lines])
                corr_pc1 = (
                    abs(np.corrcoef(scores[:, 0], conf)[0, 1]) if conf.std() > 0 else np.nan
                )
                rows_panel.append({
                    "panel": panel_name, "n_lines": line_name,
                    "n_markers": panel.n_markers,
                    "dropped_cr": audit["drop_cr"], "dropped_maf": audit["drop_maf"],
                    "missingness": audit["missing"],
                    "PC1": variance[0] * 100, "PC2": variance[1] * 100,
                    "PC3": variance[2] * 100, "corr_PC1_type": corr_pc1,
                    "bonferroni": 0.05 / panel.n_markers,
                })

                prepared = prepare_panel(
                    panel, tmp, prefix=f"panel_{index}_{line_name}",
                    kinship=vanraden_kinship(panel.dosage),
                )
                values = {trait: means[trait].reindex(lines).to_numpy(float)
                          for trait in TRAITS}

                def scan(trait: str, prepared=prepared, values=values,
                         scores=scores, index=index, line_name=line_name):
                    label = f"p{index}_{line_name}_{trait}"
                    result = run_gemma(
                        prepared, values[trait], covariates=scores[:, :3], label=label,
                    )
                    return trait, result

                with ThreadPoolExecutor(max_workers=args.jobs) as pool:
                    results = list(pool.map(scan, TRAITS))

                for trait, result in results:
                    assoc = result["assoc"]
                    summary = scan_summary(assoc, panel.n_markers)
                    rows_scan.append({
                        "panel": panel_name, "n_lines": line_name, "trait": trait,
                        "model": "EMMAX", "n_markers": panel.n_markers,
                        "lam": summary["lambda"],
                        "n_sig_5e8": summary["n_p<5e-8"],
                        "n_sig_bonf": summary["n_bonf"], "min_p": summary["min_p"],
                    })
                    hits = top_hits(assoc, 10)
                    for rank, hit in hits.iterrows():
                        rows_hits.append({
                            "panel": panel_name, "n_lines": line_name,
                            "trait": trait, "model": "EMMAX", "rank": rank + 1,
                            **hit.to_dict(),
                        })
                    if loci is not None:
                        table = loci_table(assoc, loci)
                        for _, hit in table.iterrows():
                            rows_loci.append({
                                "panel": panel_name, "n_lines": line_name,
                                "trait": trait, "model": "EMMAX", **hit.to_dict(),
                            })
                print(
                    f"{panel_name:18s} {line_name} m={panel.n_markers:6d} "
                    f"miss={audit['missing'] * 100:5.2f}% PC1={variance[0] * 100:5.2f}%",
                    flush=True,
                )

    write_csv(pd.DataFrame(rows_panel), "lines_gwas_panels.csv")
    write_csv(pd.DataFrame(rows_scan), "lines_gwas_panels_scans.csv")
    write_csv(pd.DataFrame(rows_hits), "lines_gwas_panels_hits.csv")
    if loci is not None:
        write_csv(pd.DataFrame(rows_loci), "lines_gwas_panels_loci.csv")
    print(f"\nwritten to {TABLES}")


if __name__ == "__main__":
    main()
