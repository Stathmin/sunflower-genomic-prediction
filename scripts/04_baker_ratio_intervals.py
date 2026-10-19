"""Profile-likelihood intervals for Baker's ratio.

The constrained fits are R/sommer models; the interval is the set of ratios
whose AIC is within 3.841 of the profile maximum (a 95% profile interval).
"""
from __future__ import annotations

import argparse

from sunflower.io import read_parquet, write_json
from sunflower.variance import (
    LABELS,
    baker_stats,
    hybrid_components,
    profile_all,
    shares_and_se,
)

TRAITS = ["seed_yield", "oil_content", "seed_weight_1000"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=4, help="parallel R fits")
    parser.add_argument("--step", type=float, default=0.05, help="profile grid step")
    args = parser.parse_args()

    plot = read_parquet("hybrid_plot_level.parquet").rename(columns={"father": "line"})
    fit = hybrid_components(plot, TRAITS, jobs=args.jobs)
    intervals = profile_all(plot, TRAITS, jobs=args.jobs)

    out = {}
    for trait in TRAITS:
        row = fit.loc[fit["trait"] == trait].iloc[0]
        baker = baker_stats(row)
        shares, share_se = shares_and_se(row)
        interval = intervals[trait]
        s2 = {
            "line": float(row["s2_line"]),
            "tester_x_line": float(row["s2_ml"]),
            "year_x_line": float(row["s2_yl"]),
            "year_x_tester_x_line": float(row["s2_yml"]),
        }
        s2e = float(row["s2_e"])
        out[trait] = {
            "name": trait,
            "baker": baker["baker"],
            "ci_low": interval["ci_low"],
            "ci_high": interval["ci_high"],
            "shares": [float(v) for v in shares],
            "share_se": [float(v) for v in share_se],
            "labels": LABELS,
            "s2": s2,
            "s2e": s2e,
            "gamma": [value / s2e for value in s2.values()],
            "profile": interval["profile"],
        }
        print(
            f"{trait:18s} Baker = {baker['baker']:.4f}; 95% profile interval = "
            f"{interval['ci_low']:.3f}-{interval['ci_high']:.3f}"
        )

    write_json(out, "baker_ratio_intervals.json")
    print("\nsaved tables/baker_ratio_intervals.json")


if __name__ == "__main__":
    main()
