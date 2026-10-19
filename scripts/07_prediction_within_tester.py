"""Single-hybrid prediction within each tester, five-fold by paternal line."""
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
        predicted = cv_predictions(hybrids, trait, kinship, nfold=5)
        overall = float(np.corrcoef(predicted["y"], predicted["yp"])[0, 1])
        assert np.isclose(round(overall, 3), accuracy.loc[trait, "GS_accuracy_CV1"], atol=1e-3), (
            trait, overall, accuracy.loc[trait, "GS_accuracy_CV1"]
        )
        within = {}
        for tester, group in predicted.groupby("mother"):
            r, p = stats.pearsonr(group["y"], group["yp"])
            within[tester] = (float(r), float(p), len(group))
        zbar = float(np.mean([np.arctanh(value[0]) for value in within.values()]))
        tester_share = (
            predicted.groupby("mother")["y"].transform("mean").var(ddof=0)
            / predicted["y"].var(ddof=0)
        )
        row = {
            "trait": trait, "n_hybrids": len(predicted), "r_all": overall,
            "r_within_mean": float(np.tanh(zbar)),
            "between_tester_variance_share": float(tester_share),
        }
        for tester in TESTERS:
            row[f"r_{tester}"], row[f"p_{tester}"], row[f"n_{tester}"] = within[tester]
        rows.append(row)
        text.append(
            f"{trait:17s} n={len(predicted)}  overall r={overall:.3f}  within tester: "
            + ", ".join(
                f"{tester} {within[tester][0]:.3f} (n={within[tester][2]}, "
                f"p={within[tester][1]:.2g})" for tester in TESTERS
            )
            + f"  | Fisher mean {np.tanh(zbar):.3f} | between-tester share "
            f"{tester_share * 100:.1f} %"
        )

    write_csv(pd.DataFrame(rows), "prediction_within_tester_5fold.csv")
    (TABLES / "prediction_within_tester_5fold.txt").write_text("\n".join(text), encoding="utf-8")
    print("\n".join(text))


if __name__ == "__main__":
    main()
