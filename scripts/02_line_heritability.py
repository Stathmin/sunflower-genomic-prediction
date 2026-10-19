"""Paternal-line heritability with the separated Year x Genotype term.

The variance components are fitted by R/sommer; the table keeps the published
specification (Genotype only) and the corrected specification (Genotype +
Genotype x Year) side by side.
"""
from __future__ import annotations

import pandas as pd
from scipy import stats

from sunflower.io import read_parquet, write_csv
from sunflower.paths import TABLES
from sunflower.variance import line_heritability

CONF = ("LI29", "LI30")
TRAITS = [
    "plant_height", "head_diameter", "autofertility_self", "autofertility_open",
    "days_emergence_flowering", "days_central_to_side", "hull_content",
    "seed_weight_1000", "oil_content",
]


def main() -> None:
    data = read_parquet("line_phenotypes.parquet")
    data = data[~data["genotype"].isin(CONF)]
    fit = line_heritability(data, TRAITS, jobs=4)

    rows = []
    for _, r in fit.iterrows():
        ny, ru = float(r["n_years"]), float(r["r_unit"])
        published = float(r["s2g_pub"]) / (float(r["s2g_pub"]) + float(r["s2e_pub"]) / ny)
        identifiable = ru > 1.05
        row = {
            "trait": r["trait"], "n": int(r["n"]), "n_lines": int(r["n_lines"]),
            "n_years": int(ny), "records_per_line_year": ru,
            "s2g_published": r["s2g_pub"], "s2e_published": r["s2e_pub"],
            "H2_published": published, "identifiable": identifiable,
        }
        if identifiable:
            lr = float(r["aic_pub"]) - float(r["aic_full"])
            row.update(
                s2g=r["s2g"], s2gy=r["s2gy"], s2e=r["s2e"], LR=lr,
                p_LR=0.5 * stats.chi2.sf(lr, 1) if lr > 0 else 1.0,
                H2_corrected=float(r["s2g"]) / (
                    float(r["s2g"]) + float(r["s2gy"]) / ny + float(r["s2e"]) / (ny * ru)
                ),
            )
        rows.append(row)

    table = pd.DataFrame(rows)
    write_csv(table, "line_heritability.csv")

    final = table["H2_corrected"].where(table["H2_corrected"].notna(), table["H2_published"])
    lines = ["FIX D2 - paternal-line H2 (52 oilseed, 2021-2024)", "",
             f"{'trait':24s}{'records/line-year':>16s}{'H2 publ.':>11s}{'LR':>10s}"
             f"{'p':>12s}{'H2 corr.':>11s}"]
    for _, r in table.iterrows():
        lr = f"{r['LR']:.1f}" if pd.notna(r.get("LR")) else "-"
        p = f"{r['p_LR']:.2e}" if pd.notna(r.get("p_LR")) else "-"
        h2 = f"{r['H2_corrected']:.3f}" if pd.notna(r.get("H2_corrected")) else "(unchanged)"
        lines.append(
            f"{r['trait']:24s}{r['records_per_line_year']:16.2f}"
            f"{r['H2_published']:11.3f}{lr:>10s}{p:>12s}{h2:>15s}"
        )
    lines += [
        "",
        f"published range:  {table['H2_published'].min():.3f}-{table['H2_published'].max():.3f}",
        f"corrected range:  {final.min():.3f}-{final.max():.3f}",
        "",
        "The lower bound is husk content, which has one record per line-year and the model",
        "is specified correctly; the upper bound is height, for which the Year x Genotype term is separated.",
        "Direction of the correction: upward, i.e. the published values are underestimated.",
    ]
    (TABLES / "line_heritability.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
