"""Consolidated check of numbers reported in the manuscript.

Variance components come from R/sommer, the hybrid association scan from GEMMA;
the power and GS-accuracy checks are unchanged.
"""
from __future__ import annotations

import tempfile

import numpy as np
import pandas as pd
from scipy import optimize, stats

from sunflower.external import prepare_panel, run_gemma
from sunflower.genotypes import (
    Panel,
    filter_panel,
    hybrid_dosage,
    load_genotypes,
    standardized_pca,
    vanraden_kinship,
)
from sunflower.io import read_csv, read_parquet, write_csv
from sunflower.names import en_table
from sunflower.prediction import CONF, build_hybrids, xmat
from sunflower.variance import (
    baker_stats,
    derived_components,
    hybrid_components,
)

rows = []


def check(name, claimed, got, tol=0.015) -> None:
    ok = (claimed is None) or abs(got - claimed) <= tol * max(1e-9, abs(claimed))
    rows.append({
        "value": name, "in_text": claimed, "recomputed": round(float(got), 4),
        "matched": "—" if claimed is None else ("YES" if ok else "NO"),
    })
    flag = "" if claimed is None else ("  OK" if ok else "  MISMATCH")
    print(f"  {name:52s} text={claimed!s:>8}  calc={got:.4f}{flag}")


def baker_pooled(table: pd.DataFrame) -> float:
    values = table.values
    grand = np.nanmean(values)
    father = np.nanmean(values, axis=1) - grand
    mother = np.nanmean(values, axis=0) - grand
    sca = values - grand - father[:, None] - mother[None, :]
    vg = np.nanvar(father, ddof=1)
    vs = np.nanvar(sca, ddof=1)
    return float(2 * vg / (2 * vg + vs))


def main() -> None:
    plot = read_parquet("hybrid_plot_level.parquet").rename(columns={"father": "line"})
    traits = ["seed_yield", "oil_content", "seed_weight_1000"]
    fit = hybrid_components(plot, traits, jobs=4)

    print("\n1. HERITABILITY AND REPEATABILITY")
    for trait, claim in (("oil_content", 0.90), ("seed_weight_1000", 0.91), ("seed_yield", 0.43)):
        row = fit.loc[fit["trait"] == trait].iloc[0]
        d = plot[plot["trait"] == trait]
        derived = derived_components(
            row, float(d["year"].nunique()),
            float(d.groupby(["year", "mother", "line"]).size().mean()),
        )
        check(f"H2 of the combination mean - {trait}", claim, derived["H2"], tol=0.06)
        if trait == "seed_yield":
            check("plot repeatability - seed yield", 0.16, derived["plot_repeatability"], tol=0.20)

    print("\n2. BAKER'S RATIO")
    claims = {"seed_yield": 0.440, "oil_content": 0.958, "seed_weight_1000": 0.951}
    table_claims = {"seed_yield": 0.716, "oil_content": 0.936, "seed_weight_1000": 0.936}
    delivered = en_table(read_parquet("hybrids_as_delivered.parquet"))
    mother_column = "mother_long" if "mother_long" in delivered.columns else "mother"
    for trait in traits:
        row = fit.loc[fit["trait"] == trait].iloc[0]
        check(f"baker {trait} (sommer)", claims[trait], baker_stats(row)["baker"], tol=0.03)
        pooled = delivered[delivered["trait"] == trait].groupby(
            ["father", mother_column]
        )["value"].mean().unstack()
        check(
            f"baker {trait} (year-means table, robustness)",
            table_claims[trait], baker_pooled(pooled), tol=0.03,
        )

    print("\n3. MATERNAL-LINE GCA")
    mothers = en_table(read_csv("gca_testers_as_delivered.csv"))
    mothers = mothers.set_index("mother_long")
    for mother, column, claim in (
        ("VK101A", "oil_content", 1.20), ("VK934A", "seed_weight_1000", 10.9),
        ("VK934A", "plant_height", 8.4),
    ):
        if column in mothers.columns and mother in mothers.index:
            check(f"GCA {mother} - {column}", claim, float(mothers.loc[mother, column]), tol=0.05)

    print("\n4. POWER ANALYSIS (alpha=5e-8)")
    critical = stats.chi2.isf(5e-8, 1)

    def power(n, pve):
        return stats.ncx2.sf(critical, 1, n * pve / (1 - pve))

    check(
        "detectable PVE at n=54 and 80% power", 42.0,
        optimize.brentq(lambda p: power(54, p) - 0.8, 0.01, 0.99) * 100, tol=0.03,
    )
    for pve, claim in ((0.15, 225), (0.10, 355), (0.05, 750)):
        check(
            f"n for 80% power at PVE {pve * 100:.0f} %", claim,
            optimize.brentq(lambda n, pve=pve: power(n, pve) - 0.8, 10, 20000), tol=0.03,
        )

    print("\n5. HYBRID ASSOCIATION SCAN (GEMMA)")
    data = load_genotypes()
    panel57 = filter_panel(data, call_rate=0.9, maf=0.05)
    blup = read_csv("hybrid_blup_phenotypes.csv")
    hybrids = build_hybrids(blup, panel57.samples, conf=CONF)
    hybrids = hybrids.dropna(subset=["seed_weight_1000"]).reset_index(drop=True)
    hg = hybrid_dosage(
        panel57.dosage, panel57.samples,
        hybrids["mother"].tolist(), hybrids["father"].tolist(),
    )
    hybrid_panel = Panel(
        dosage=hg.T, raw=hg.T, chrom=panel57.chrom, pos=panel57.pos,
        samples=tuple(hybrids["hybrid"]), mask=panel57.mask, missing_fraction=0.0,
    )
    scores, _ = standardized_pca(hg.T, 3)
    design = xmat(hybrids, "mother")
    with tempfile.TemporaryDirectory(prefix="sunflower-hybrid-") as tmp:
        prepared = prepare_panel(
            hybrid_panel, tmp, prefix="hybrid",
            kinship=vanraden_kinship(hybrid_panel.dosage),
        )
        for label, covariates in (
            ("mothers", design[:, 1:]),
            ("mothers_plus_3PC", np.column_stack([design[:, 1:], scores])),
        ):
            result = run_gemma(
                prepared, hybrids["seed_weight_1000"].to_numpy(float),
                covariates=covariates, label=f"hybrid_{label}",
            )
            p = result["assoc"]["p"].to_numpy(float)
            valid = np.isfinite(p)
            lam = float(np.median(stats.chi2.isf(p[valid], 1)) / stats.chi2.ppf(0.5, 1))
            check(f"hybrid EMMAX lambda ({label})", None, lam)
            check(
                f"significant SNP at 5e-8, hybrids ({label})",
                0, int(np.sum(p[valid] < 5e-8)), tol=0,
            )

    print("\n6. GENOMIC-SELECTION ACCURACY")
    accuracy = read_csv("prediction_accuracy_5fold.csv", comment="#").set_index("trait")
    for trait, claim in (
        ("seed_weight_1000", 0.84), ("oil_content", 0.65),
        ("oil_yield", 0.35), ("seed_yield", 0.29),
    ):
        check(f"GS accuracy - {trait}", claim, float(accuracy.loc[trait, "GS_accuracy_CV1"]), tol=0.03)

    result = pd.DataFrame(rows)
    write_csv(result, "reported_numbers_check.csv")
    bad = result[result["matched"] == "NO"]
    print(f"\nchecks: {len(result)}; matched: {(result['matched'] == 'YES').sum()}; "
          f"mismatches: {len(bad)}; informational: {(result['matched'] == '—').sum()}")
    if len(bad):
        print(bad.to_string(index=False))


if __name__ == "__main__":
    main()
