"""Sensitivity of paternal-GCA prediction to BLUP refitting and outlier removal."""
from __future__ import annotations

import numpy as np
import pandas as pd

from sunflower.genotypes import filter_panel, load_genotypes, vanraden_kinship
from sunflower.io import read_csv, read_parquet, write_csv
from sunflower.prediction import CONF, gblup_cv, hybrid_blups

TR3 = ["seed_yield", "oil_content", "seed_weight_1000"]
TR4 = TR3 + ["oil_yield"]


def main() -> None:
    data = load_genotypes()
    panel = filter_panel(data, call_rate=0.9, maf=0.05)
    blup = read_csv("hybrid_blup_phenotypes.csv")
    blup[["mother", "father"]] = blup["hybrid"].str.split("_", n=1, expand=True)
    fathers = sorted(blup["father"].unique())
    kept = [father for father in fathers if father not in CONF]
    columns = [panel.samples.index(father) for father in fathers]
    kinship = vanraden_kinship(panel.dosage[:, columns]) + np.eye(len(fathers)) * 1e-6
    kinship = kinship[np.ix_(
        [fathers.index(father) for father in kept],
        [fathers.index(father) for father in kept],
    )]

    plots = read_parquet("hybrid_plot_level.parquet")
    plots = plots[plots["trait"].isin(TR3)].copy()

    def line_means(frame: pd.DataFrame) -> pd.DataFrame:
        means = hybrid_blups(frame, TR3)
        means["oil_yield"] = means["seed_yield"] * means["oil_content"] / 100
        means["father"] = [hybrid.split("_", 1)[1] for hybrid in means["hybrid"]]
        return means.groupby("father").mean(numeric_only=True).reindex(kept)

    observed = blup[blup["father"].isin(kept)].groupby("father")[TR4].mean().reindex(kept)
    reference = line_means(plots)
    assert np.allclose(reference[TR4].to_numpy(), observed[TR4].to_numpy(), atol=1e-6), (
        "full-data BLUPs do not reproduce the deposited table"
    )

    def lolo(means: pd.DataFrame) -> dict:
        out = {}
        for trait in TR4:
            y = means[trait].to_numpy(float)
            predicted = np.array([
                gblup_cv(y, kinship, [j for j in range(len(kept)) if j != k], [k])[0]
                for k in range(len(kept))
            ])
            out[trait] = float(np.corrcoef(predicted, y)[0, 1])
        return out

    rows = [{
        "analysis": "main (BLUPs from all plots)",
        **{trait: round(value, 3) for trait, value in lolo(reference).items()},
    }]

    predictions = {trait: np.empty(len(kept)) for trait in TR4}
    for k, father in enumerate(kept):
        means = line_means(plots[plots["father"] != father])
        train = [j for j in range(len(kept)) if j != k]
        for trait in TR4:
            y = means[trait].to_numpy(float)
            y[k] = 0.0
            predictions[trait][k] = gblup_cv(y, kinship, train, [k])[0]
        print(f"  fold {k + 1}/{len(kept)} ({father})", flush=True)
    rows.append({
        "analysis": "BLUPs refitted without the held-out line",
        **{trait: round(float(np.corrcoef(predictions[trait], observed[trait])[0, 1]), 3)
           for trait in TR4},
    })

    removed = read_csv("removed_plots.csv")
    removed = removed[removed["trait"].isin(TR3)].copy()
    full = pd.concat([plots, removed[plots.columns]], ignore_index=True)
    alternative = line_means(full)
    rows.append({
        "analysis": f"removed plot values put back (n = {len(removed)})",
        **{trait: round(value, 3) for trait, value in lolo(alternative).items()},
    })

    table = pd.DataFrame(rows)
    write_csv(table, "prediction_gca_sensitivity.csv")
    print("\nPredictive ability of paternal GCA (r), leave-one-line-out, 52 oilseed lines:")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
