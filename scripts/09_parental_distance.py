"""Reproduce the n = 162 reconciliation of parental distance and hybrid yield."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from sunflower.genotypes import filter_panel, load_genotypes
from sunflower.io import read_csv, read_parquet
from sunflower.names import en_table

warnings.filterwarnings("ignore")
TESTER_MAP = {"VK101A": "VK101", "VA761A": "VA761", "VK934A": "VK934"}


def main() -> None:
    data = load_genotypes()
    panel = filter_panel(data, call_rate=0.9, maf=0.05)
    print(f"distance panel: {panel.n_markers:,} SNP")

    plot = en_table(read_parquet("hybrid_plot_level.parquet"))
    means162 = (
        plot.groupby(["mother", "father", "trait"])["value"].mean()
        .unstack("trait").reset_index()
    )
    print("hybrid means from plot-level data:", len(means162))

    delivered = en_table(read_csv("hybrid_year_means_as_delivered.csv"))
    key159 = set(zip(delivered["mother_long"], delivered["father"]))
    key162 = set(zip(means162["mother"], means162["father"]))
    print("in plot-level data but not in the summary:", sorted(key162 - key159))

    index = {sample: i for i, sample in enumerate(panel.samples)}

    def distance(mother_long: str, father: str) -> float:
        mother = np.round(panel.dosage[:, index[TESTER_MAP[mother_long]]])
        line = np.round(panel.dosage[:, index[father]])
        return float(np.mean(np.abs(mother - line) / 2.0))

    means162["dist"] = [distance(m, f) for m, f in zip(means162["mother"], means162["father"])]

    def correlations(frame: pd.DataFrame, label: str):
        rows = []
        x, y = frame["dist"].to_numpy(), frame["seed_yield"].to_numpy()
        mask = ~np.isnan(y)
        rows.append(("ALL", int(mask.sum()), *stats.pearsonr(x[mask], y[mask])))
        for mother in ["VK934A", "VK101A", "VA761A"]:
            subset = frame[frame["mother"] == mother]
            xx, yy = subset["dist"].to_numpy(), subset["seed_yield"].to_numpy()
            keep = ~np.isnan(yy)
            rows.append((mother, int(keep.sum()), *stats.pearsonr(xx[keep], yy[keep])))
        print(f"\n--- distance x seed yield, {label} ---")
        for group, n, r, p in rows:
            print(f"  {group:8s} n={n:3d}  r={r:.4f}  p={p:.5f}")
        return rows

    sub159 = means162[
        [(m, f) in key159 for m, f in zip(means162["mother"], means162["father"])]
    ]
    correlations(means162, "n = 162 (plot-level means)")
    correlations(sub159, "n = 159 (summary-table combinations only)")

    hm2 = delivered.copy()
    hm2["dist"] = [distance(m, f) for m, f in zip(hm2["mother_long"], hm2["father"])]
    hm2 = hm2.drop(columns=["mother"]).rename(columns={"mother_long": "mother"})
    correlations(hm2, "n = 159, values from the delivered year-means table")

    def baker(frame: pd.DataFrame, trait: str) -> float:
        table = frame.pivot_table(index="father", columns="mother", values=trait)
        values = table.values
        grand = np.nanmean(values)
        father = np.nanmean(values, axis=1) - grand
        mother = np.nanmean(values, axis=0) - grand
        sca = values - grand - father[:, None] - mother[None, :]
        vg = np.nanvar(np.concatenate([father, mother]))
        vs = np.nanvar(sca.ravel())
        return float(2 * vg / (2 * vg + vs))

    dependent = means162.dropna(subset=["dist", "seed_yield"]).copy()
    dependent["zd"] = (dependent["dist"] - dependent["dist"].mean()) / dependent["dist"].std()
    mixed = smf.mixedlm(
        "seed_yield ~ zd + C(mother)", dependent, groups=dependent["father"]
    ).fit(reml=True)
    print("\n--- distance x seed yield allowing for dependence among hybrids (162 hybrids) ---")
    print(
        f"  mixed model, tester fixed, paternal line random: slope per SD = "
        f"{mixed.params['zd']:+.4f} t/ha, SE {mixed.bse['zd']:.4f}, "
        f"p = {mixed.pvalues['zd']:.2e}"
    )
    line_means = dependent.groupby("father")[["dist", "seed_yield"]].mean()
    rf, pf = stats.pearsonr(line_means["dist"], line_means["seed_yield"])
    print(
        f"  paternal-line means over the three testers (n = {len(line_means)}): "
        f"r = {rf:.3f}, p = {pf:.4f}"
    )
    ols = smf.ols("seed_yield ~ zd + C(mother) + C(father)", dependent).fit()
    print(
        f"  within paternal line and tester: slope = {ols.params['zd']:+.4f}, "
        f"p = {ols.pvalues['zd']:.3f}"
    )

    print("\n--- Baker ratio ---")
    for trait in ["seed_yield", "oil_content", "seed_weight_1000"]:
        print(
            f"  {trait:12s} 162={baker(means162, trait):.4f} "
            f"159-subset={baker(sub159, trait):.4f} "
            f"summary-159={baker(hm2, trait):.4f}"
        )

    print("\n--- GCA source ---")
    stored_fathers = en_table(read_csv("gca_fathers_as_delivered.csv")).set_index("father")
    stored_mothers = en_table(read_csv("gca_testers_as_delivered.csv"))

    def gca_from(frame: pd.DataFrame, trait: str):
        table = frame.pivot_table(index="father", columns="mother", values=trait)
        grand = np.nanmean(table.values)
        return (
            pd.Series(np.nanmean(table.values, 1) - grand, index=table.index),
            pd.Series(np.nanmean(table.values, 0) - grand, index=table.columns),
        )

    for trait in ["seed_yield", "oil_content", "seed_weight_1000"]:
        father162, _ = gca_from(means162, trait)
        fatherhm, _ = gca_from(hm2, trait)
        stored = stored_fathers[trait]
        print(
            f"  {trait:16s} corr(stored, 162)="
            f"{np.corrcoef(stored.reindex(father162.index), father162)[0, 1]:.6f} "
            f"| corr(stored, summary-159)="
            f"{np.corrcoef(stored.reindex(fatherhm.index), fatherhm)[0, 1]:.6f} "
            f"| max|diff| vs 162="
            f"{np.max(np.abs(stored.reindex(father162.index).to_numpy() - father162.to_numpy())):.4f}"
        )
    print("\n  maternal GCA, stored:")
    print(
        stored_mothers[
            ["mother_long", "seed_yield", "oil_content", "seed_weight_1000", "plant_height"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
