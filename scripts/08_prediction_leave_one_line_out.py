"""Single-hybrid prediction with leave-one-father-out cross-validation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from sunflower.genotypes import filter_panel, load_genotypes
from sunflower.io import read_csv, write_csv
from sunflower.paths import TABLES
from sunflower.prediction import CONF, TESTERS, build_hybrids, cv_predictions, hybrid_kinship

TRAITS = ["seed_weight_1000", "oil_content", "oil_yield", "seed_yield"]


def main() -> None:
    data = load_genotypes()
    panel = filter_panel(data, call_rate=0.9, maf=0.05)
    blup = read_csv("hybrid_blup_phenotypes.csv")
    hybrids = build_hybrids(blup, panel.samples, conf=CONF)
    kinship = hybrid_kinship(
        panel.dosage, panel.samples,
        hybrids["mother"].tolist(), hybrids["father"].tolist(),
    )
    accuracy = read_csv("prediction_accuracy_5fold.csv", comment="#").set_index("trait")

    rows, text = [], [f"SNPs: {panel.n_markers:,}; hybrids with genotypes: {len(hybrids)}", ""]
    for trait in TRAITS:
        five = cv_predictions(hybrids, trait, kinship, nfold=5)
        r5 = float(np.corrcoef(five["y"], five["yp"])[0, 1])
        assert np.isclose(round(r5, 3), accuracy.loc[trait, "GS_accuracy_CV1"], atol=1e-3), (
            trait, r5, accuracy.loc[trait, "GS_accuracy_CV1"]
        )
        n_fathers = hybrids.dropna(subset=[trait])["father"].nunique()
        predicted = cv_predictions(hybrids, trait, kinship, nfold=n_fathers)
        assert predicted.groupby("father").ngroups == n_fathers

        overall = float(np.corrcoef(predicted["y"], predicted["yp"])[0, 1])
        within = {}
        for tester, group in predicted.groupby("mother"):
            r, p = stats.pearsonr(group["y"], group["yp"])
            within[tester] = (float(r), float(p), len(group))
        zbar = float(np.mean([np.arctanh(value[0]) for value in within.values()]))
        row = {
            "trait": trait, "n_hybrids": len(predicted), "n_fathers": n_fathers,
            "r_all_lofo": overall, "r_within_mean_lofo": float(np.tanh(zbar)),
            "r_all_5fold_control": r5,
        }
        for tester in TESTERS:
            row[f"r_{tester}"], row[f"p_{tester}"], row[f"n_{tester}"] = within[tester]
        rows.append(row)
        text.append(
            f"{trait:12s} n={len(predicted)} fathers={n_fathers}  LOFO overall r={overall:.3f}, "
            "within tester " + ", ".join(
                f"{tester} {within[tester][0]:.3f}" for tester in TESTERS
            )
            + f", Fisher mean {np.tanh(zbar):.3f} | control 5-fold r={r5:.3f}"
        )

    write_csv(pd.DataFrame(rows), "prediction_leave_one_line_out.csv")
    (TABLES / "prediction_leave_one_line_out.txt").write_text("\n".join(text), encoding="utf-8")
    print("\n".join(text))


if __name__ == "__main__":
    main()
