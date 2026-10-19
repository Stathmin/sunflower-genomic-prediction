"""Variance components, Baker's ratio and H2 of the combination mean.

The REML fit is delegated to R/sommer; this script only assembles the table.
"""
from __future__ import annotations

import pandas as pd

from sunflower.io import read_parquet, write_csv
from sunflower.paths import TABLES
from sunflower.variance import (
    baker_stats,
    derived_components,
    hybrid_components,
    moments_baker,
)

TRAITS = ["seed_yield", "oil_content", "seed_weight_1000", "plant_height", "head_diameter"]


def main() -> None:
    plot = read_parquet("hybrid_plot_level.parquet").rename(columns={"father": "line"})
    fit = hybrid_components(plot, TRAITS, jobs=4)

    rows, text = [], []
    for _, r in fit.iterrows():
        trait = r["trait"]
        d = plot[plot["trait"] == trait]
        ny = float(d["year"].nunique())
        r_plots = float(d.groupby(["year", "mother", "line"]).size().mean())
        baker = baker_stats(r)
        derived = derived_components(r, ny, r_plots)
        rows.append({
            "trait": trait, "n": int(r["n"]),
            "n_combinations": d.groupby(["mother", "line"]).ngroups,
            "plots_per_combination": r_plots,
            "s2_line": r["s2_line"], "s2_tester_x_line": r["s2_ml"],
            "s2_year_x_line": r["s2_yl"], "s2_year_x_tester_x_line": r["s2_yml"],
            "s2_error": r["s2_e"],
            "baker_REML": baker["baker"], "se": baker["se_baker"],
            "baker_moments": moments_baker(plot, trait),
            "plot_repeatability": derived["plot_repeatability"],
            "H2_of_combination_mean": derived["H2"],
        })
        components = {
            "Line": r["s2_line"], "Tester x Line": r["s2_ml"],
            "Year x Line": r["s2_yl"], "Year x Tester x Line": r["s2_yml"],
            "error": r["s2_e"],
        }
        total = sum(components.values())
        text.append(
            f"### {trait}: n={int(r['n'])} plots, {rows[-1]['n_combinations']} combinations, "
            f"r={r_plots:.3f}"
        )
        text.append(
            f"    Baker's ratio = {baker['baker']:.4f} +- {baker['se_baker']:.4f}"
            f"   (moments {rows[-1]['baker_moments']:.4f})"
        )
        text.append(
            "    shares: " + "  ".join(f"{name} {value / total * 100:5.1f} %"
                                      for name, value in components.items())
        )
        text.append(
            f"    plot repeatability = {derived['plot_repeatability']:.4f};  "
            f"H2 of combination mean = {derived['H2']:.4f}"
        )
        text.append("")

    write_csv(pd.DataFrame(rows), "variance_components.csv")
    (TABLES / "variance_components.txt").write_text("\n".join(text), encoding="utf-8")
    print("\n".join(text))


if __name__ == "__main__":
    main()
