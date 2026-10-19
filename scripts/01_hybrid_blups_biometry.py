"""Hybrid BLUP phenotypes for height and diameter, plus the published columns.

The model is the same as for the productivity traits:
    value ~ C(year-plot), combination random, REML;
    BLUP = grand mean + combination random effect.
"""
from __future__ import annotations

import numpy as np

from sunflower.io import read_csv, read_parquet, write_csv
from sunflower.paths import TABLES
from sunflower.prediction import hybrid_blups

TRAITS = ["plant_height", "head_diameter", "seed_yield", "oil_content", "seed_weight_1000"]


def main() -> None:
    plot = read_parquet("hybrid_plot_level.parquet")
    published = read_csv("hybrid_blup_phenotypes.csv")
    new, stats = hybrid_blups(plot, TRAITS, return_stats=True)

    lines = ["Hybrid BLUP phenotypes: extension to height and diameter", ""]
    for trait in TRAITS:
        s = stats[trait]
        lines.append(
            f"{trait:14s} n={s['n']:4d} combinations={s['combinations']:3d}  "
            f"s2g={s['s2g']:8.3f} s2e={s['s2e']:8.3f}  "
            f"repeatability={s['repeatability']:.3f}  H2(mean)={s['H2']:.3f}"
        )
        if trait not in ("plant_height", "head_diameter"):
            check = published.set_index("hybrid")[trait].reindex(new["hybrid"])
            values = new.set_index("hybrid")[trait]
            delta = (check - values).abs()
            corr = np.corrcoef(check.dropna(), values[check.dropna().index])[0, 1]
            lines.append(
                f"{'':14s} check vs published column: max |delta| = {delta.max():.4f} "
                f"(correlation {corr:.6f})"
            )

    extended = published.set_index("hybrid").copy()
    for trait in ("plant_height", "head_diameter"):
        extended[trait] = new.set_index("hybrid")[trait]
    extended = extended.reset_index()
    write_csv(extended, "hybrid_blup_phenotypes_with_biometry.csv")

    lines += [
        "",
        f"written: tables/hybrid_blup_phenotypes_with_biometry.csv, "
        f"{len(extended)} hybrids x {extended.shape[1] - 1} traits",
        f"columns: {', '.join(extended.columns[1:])}",
        f"non-missing in new columns: plant_height {extended['plant_height'].notna().sum()}, "
        f"head_diameter {extended['head_diameter'].notna().sum()}",
    ]
    (TABLES / "hybrid_blup_biometry.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
