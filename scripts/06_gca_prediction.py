"""Genomic prediction of paternal GCA.

Sections: effective n, leave-one-father-out GCA, learning curve, repeated
five-fold CV, and the published predictive-ability scale.  The GBLUP solver is
shared (``sunflower.prediction``) and cross-checked against R/rrBLUP in the test
suite.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sunflower.genotypes import filter_panel, load_genotypes, vanraden_kinship
from sunflower.io import read_csv, write_csv
from sunflower.prediction import CONF, build_hybrids, gblup_cv

TRAITS = ["seed_yield", "oil_content", "seed_weight_1000", "oil_yield"]


def main() -> None:
    data = load_genotypes()
    panel = filter_panel(data, call_rate=0.9, maf=0.05)
    blup_all = read_csv("hybrid_blup_phenotypes.csv")
    blup = build_hybrids(blup_all, data.samples, conf=())
    fathers = sorted(blup["father"].unique())
    columns = [panel.samples.index(father) for father in fathers]
    kinship = vanraden_kinship(panel.dosage[:, columns]) + np.eye(len(fathers)) * 1e-6

    print("=" * 86)
    print("A. EFFECTIVE n")
    print("=" * 86)
    print(f"  hybrids: {blup['hybrid'].nunique()}")
    print(f"  paternal lines: {blup['father'].nunique()}; testers: {blup['mother'].nunique()}")
    print(f"  panel for K: {panel.dosage.shape[0]:,} SNP, {len(fathers)} fathers")

    print("\n" + "=" * 86)
    print("B. PATERNAL GCA: leave-one-father-out")
    print("=" * 86)
    loo = {}
    for trait in TRAITS:
        means = blup.dropna(subset=[trait]).groupby("father")[trait].mean()
        lines = [f for f in fathers if f in means.index and f not in CONF]
        y = means.reindex(lines).to_numpy(float)
        index = [fathers.index(line) for line in lines]
        k = kinship[np.ix_(index, index)]
        predicted = np.array([
            gblup_cv(y, k, [j for j in range(len(y)) if j != i], [i])[0]
            for i in range(len(y))
        ])
        r = float(np.corrcoef(predicted, y)[0, 1])
        loo[trait] = (len(lines), r)
        print(f"  {trait:14s} fathers {len(lines):3d}   r(predicted, observed) = {r:+.3f}")
    write_csv(
        pd.DataFrame([
            {"trait": trait, "n_lines": n, "r": r} for trait, (n, r) in loo.items()
        ]),
        "prediction_gca_leave_one_out.csv",
    )

    print("\n" + "=" * 86)
    print("C. LEARNING CURVE")
    print("=" * 86)
    sizes = [10, 20, 30, 40, 45, 50]
    repeats = 40
    rng = np.random.default_rng(20260818)
    rows = []
    print(f"  {'trait':14s} " + "  ".join(f"n={s:<3d}" for s in sizes) + "   LOO")
    for trait in TRAITS:
        means = blup.dropna(subset=[trait]).groupby("father")[trait].mean()
        lines = [f for f in fathers if f in means.index and f not in CONF]
        y = means.reindex(lines).to_numpy(float)
        index = [fathers.index(line) for line in lines]
        k = kinship[np.ix_(index, index)]
        summary = []
        for size in sizes:
            values = []
            for _ in range(repeats):
                perm = rng.permutation(len(lines))
                train, test = perm[:size], perm[size:]
                if len(test) < 5:
                    continue
                predicted = gblup_cv(y, k, list(train), list(test))
                if np.std(predicted) < 1e-12:
                    continue
                values.append(np.corrcoef(predicted, y[test])[0, 1])
            summary.append(np.mean(values) if values else np.nan)
            if values:
                rows.append({
                    "trait": trait, "n_train": size, "n_test": len(lines) - size,
                    "n_draws": len(values), "r_mean": np.mean(values),
                    "r_sd": np.std(values, ddof=1),
                    "r_q025": np.quantile(values, 0.025),
                    "r_q975": np.quantile(values, 0.975),
                })
        print(f"  {trait:14s} " + "  ".join(f"{v:+.3f}" for v in summary)
              + f"   {loo[trait][1]:+.3f}")
    write_csv(pd.DataFrame(rows), "learning_curve.csv")

    print("\n" + "=" * 86)
    print("C2. REPEATED FIVE-FOLD CV OF PATERNAL GCA (50 repeats)")
    print("=" * 86)
    rng5 = np.random.default_rng(20261003)
    rows = []
    for trait in TRAITS:
        means = blup.dropna(subset=[trait]).groupby("father")[trait].mean()
        lines = [f for f in fathers if f in means.index and f not in CONF]
        y = means.reindex(lines).to_numpy(float)
        index = [fathers.index(line) for line in lines]
        k = kinship[np.ix_(index, index)]
        values = []
        for _ in range(50):
            perm = rng5.permutation(len(lines))
            predicted = np.empty(len(lines))
            for fold in np.array_split(perm, 5):
                held = set(fold.tolist())
                predicted[fold] = gblup_cv(
                    y, k, [j for j in perm if j not in held], list(fold),
                )
            values.append(np.corrcoef(predicted, y)[0, 1])
        rows.append({
            "trait": trait, "n_lines": len(lines), "repeats": 50,
            "r_mean": np.mean(values), "r_sd": np.std(values, ddof=1),
            "r_q025": np.quantile(values, 0.025), "r_q975": np.quantile(values, 0.975),
        })
        print(f"  {trait:14s} lines {len(lines):3d}   r = {np.mean(values):.3f} "
              f"+- {np.std(values, ddof=1):.3f}")
    write_csv(pd.DataFrame(rows), "prediction_gca_repeated_5fold.csv")

    print("\n" + "=" * 86)
    print("D. RESCALE TO ACCURACY = r / sqrt(H2)")
    print("=" * 86)
    accuracy = read_csv("prediction_accuracy_5fold.csv", comment="#")
    for _, row in accuracy.iterrows():
        h2 = row["H2_of_combination_mean"]
        if pd.notna(h2):
            print(f"  {row['trait']:14s} predictive ability {row['GS_accuracy_CV1']:.3f}"
                  f" | H2 {h2:.3f} | accuracy {row['GS_accuracy_CV1'] / np.sqrt(h2):.3f}")
        else:
            print(f"  {row['trait']:14s} predictive ability {row['GS_accuracy_CV1']:.3f}"
                  " | H2 not estimated")


if __name__ == "__main__":
    main()
