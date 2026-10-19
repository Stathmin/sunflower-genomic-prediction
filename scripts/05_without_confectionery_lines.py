"""Variance components and Baker's ratio without the two confectionery lines.

The full set is kept as a control.  The control comparison uses the deposited
tables with a tolerance because the REML engine is now R/sommer rather than the
original bespoke implementation (the Baker ratios agree to ~1e-4).
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from sunflower.io import read_csv, read_json, read_parquet, write_csv
from sunflower.paths import TABLES
from sunflower.variance import (
    baker_stats,
    derived_components,
    hybrid_components,
    moments_baker,
    profile_interval,
)

CONF = ("LI29", "LI30")
TRAITS = ["seed_yield", "oil_content", "seed_weight_1000", "plant_height", "head_diameter"]
WITH_CI = ("seed_yield", "oil_content", "seed_weight_1000")


def _row(row: pd.Series, data: pd.DataFrame, with_ci: bool, plot: pd.DataFrame) -> dict:
    baker = baker_stats(row)
    ny = float(data["year"].nunique())
    r = float(data.groupby(["year", "mother", "line"]).size().mean())
    derived = derived_components(row, ny, r)
    out = {
        "n": int(row["n"]),
        "n_combinations": data.groupby(["mother", "line"]).ngroups,
        "n_lines": data["line"].nunique(),
        "plots_per_combination": r,
        "s2_line": row["s2_line"], "s2_tester_x_line": row["s2_ml"],
        "s2_year_x_line": row["s2_yl"], "s2_year_x_tester_x_line": row["s2_yml"],
        "s2_error": row["s2_e"],
        "gca_share": derived["gca_share"],
        "baker": baker["baker"], "se_baker": baker["se_baker"],
        "baker_moments": moments_baker(plot, row["trait"]),
        "plot_repeatability": derived["plot_repeatability"],
        "H2": derived["H2"],
        "ci_low": np.nan, "ci_high": np.nan,
    }
    if with_ci:
        interval = profile_interval(plot, row["trait"])
        out.update(ci_low=interval["ci_low"], ci_high=interval["ci_high"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()

    plot = read_parquet("hybrid_plot_level.parquet").rename(columns={"father": "line"})
    reduced = plot[~plot["line"].isin(CONF)]
    full_fit = hybrid_components(plot, TRAITS, jobs=args.jobs)
    reduced_fit = hybrid_components(reduced, TRAITS, jobs=args.jobs)

    reference = read_csv("variance_components.csv").set_index("trait")
    intervals = read_json("baker_ratio_intervals.json")

    rows, text = [], []
    for trait in TRAITS:
        full_row = full_fit.loc[full_fit["trait"] == trait].iloc[0]
        reduced_row = reduced_fit.loc[reduced_fit["trait"] == trait].iloc[0]
        a = _row(full_row, plot[plot["trait"] == trait], trait in WITH_CI, plot)
        b = _row(reduced_row, reduced[reduced["trait"] == trait], trait in WITH_CI, reduced)

        # Control: the new engine should reproduce the deposited table closely.
        assert np.isclose(a["baker"], reference.loc[trait, "baker_REML"], atol=1e-3), trait
        assert np.isclose(a["H2"], reference.loc[trait, "H2_of_combination_mean"], atol=1e-3), trait
        if trait in WITH_CI:
            assert np.isclose(a["ci_low"], intervals[trait]["ci_low"], atol=1e-2), trait
            assert np.isclose(a["ci_high"], intervals[trait]["ci_high"], atol=1e-2), trait

        for label, values in (("all 162 combinations (control)", a),
                              ("without LI29 and LI30", b)):
            rows.append({"trait": trait, "set": label, **values})
        text += [
            f"### {trait}",
            f"  control: baker {a['baker']:.4f} +/- {a['se_baker']:.4f} "
            f"[{a['ci_low']:.3f}-{a['ci_high']:.3f}]  GCA share {a['gca_share'] * 100:.1f} %  "
            f"H2 {a['H2']:.4f}",
            f"  without LI29/LI30: {b['n_combinations']} combinations, {b['n_lines']} lines, "
            f"{b['n']} plots; baker {b['baker']:.4f} +/- {b['se_baker']:.4f} "
            f"[{b['ci_low']:.3f}-{b['ci_high']:.3f}]  GCA share {b['gca_share'] * 100:.1f} %  "
            f"H2 {b['H2']:.4f}  plot repeatability {b['plot_repeatability']:.4f}",
            "",
        ]

    write_csv(pd.DataFrame(rows), "without_confectionery_lines.csv")
    (TABLES / "without_confectionery_lines.txt").write_text("\n".join(text), encoding="utf-8")
    print("\n".join(text))


if __name__ == "__main__":
    main()
