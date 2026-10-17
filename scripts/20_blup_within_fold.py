# -*- coding: utf-8 -*-
"""Two sensitivity checks of the leave-one-line-out prediction of paternal GCA (Section 3.2, Section 4.5).

1. Hybrid BLUPs recomputed inside each fold. In the main analysis the hybrid BLUPs (tables/hybrid_blup_phenotypes.csv)
   are computed once from all plots, so the plots of a held-out line contribute to the year and replicate effects of the
   training data. Here, for each held-out paternal line, the BLUPs are refitted from the plots of the other lines only,
   the line means of the training lines are taken from these BLUPs, and the held-out line is predicted by GBLUP as in
   scripts/06_gca_prediction.py. The observed value of the held-out line is its line mean from the full-data BLUPs.
2. Outlier screening. The nine plot values removed by the four-standard-deviation rule (Section 2.1) are put back and the
   hybrid BLUPs and the leave-one-line-out prediction are recomputed.

Reads:  tables/hybrid_plot_level.parquet, tables/hybrid_blup_phenotypes.csv, genotypes/merged_genotypes.npz,
        tables/removed_plots.csv (the nine removed plot values, from the companion field-study repository)
Writes: tables/prediction_gca_sensitivity.csv, logs via stdout
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from scipy.optimize import minimize_scalar

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from paths import OUT, DEPOSIT
NPZ = DEPOSIT / "merged_genotypes.npz"
from names import en_table

CONF = {"LI29", "LI30"}
TR3 = ["seed_yield", "oil_content", "seed_weight_1000"]
TR4 = TR3 + ["oil_yield"]

blup = en_table(pd.read_csv(OUT / "hybrid_blup_phenotypes.csv"))
blup[["mother", "father"]] = blup["hybrid"].str.split("_", n=1, expand=True)
P = en_table(pd.read_parquet(OUT / "hybrid_plot_level.parquet"))
P = P[P["trait"].isin(TR3)].copy()
P["g"] = P["mother"] + "_" + P["father"]
P["yr_rep"] = P["year"].astype(str) + "_" + P["rep"].astype(str)
fathers = sorted(blup["father"].unique())
fl = [f for f in fathers if f not in CONF]

# ---- kinship of the paternal lines, as in script 06
z = np.load(NPZ, allow_pickle=True)
G = z["genotypes"].astype(float); G[G < 0] = np.nan
from names import en_ids
samp = en_ids(z["samples"].astype(str)); idx = {s: i for i, s in enumerate(samp)}
chrom = np.array([int(str(c).replace("CM00", "").replace(".2", "")) - 7889 if str(c).startswith("CM00") else -1 for c in z["chrom"]])
with np.errstate(all="ignore"):                 # filters over all 57 sequenced lines, as in script 06
    cr = np.mean(~np.isnan(G), 1); af = np.nanmean(G, 1) / 2
keep = (chrom >= 1) & (chrom <= 17) & (cr >= 0.9) & (np.minimum(af, 1 - af) >= 0.05)
M = G[keep]; M = np.where(np.isnan(M), np.nanmean(M, 1, keepdims=True), M)
Mf = M[:, [idx[f] for f in fathers]]            # kinship over the 54 paternal lines, as in script 06
p = Mf.mean(1) / 2; Zc = (Mf - 2 * p[:, None]).T
K54 = (Zc @ Zc.T) / (2 * np.sum(p * (1 - p))) + np.eye(len(fathers)) * 1e-6
K = K54[np.ix_([fathers.index(f) for f in fl], [fathers.index(f) for f in fl])]


def gblup_cv(y, K, train_idx, test_idx):
    ytr = y[train_idx]; Ktr = K[np.ix_(train_idx, train_idx)]; ytr = ytr - ytr.mean(); n = len(ytr)
    def negll(loglam):
        lam = np.exp(loglam); V = Ktr + np.eye(n) / lam
        _, ld = np.linalg.slogdet(V); a = np.linalg.solve(V, ytr)
        return 0.5 * (ld + n * np.log(ytr @ a))
    lam = np.exp(minimize_scalar(negll, bounds=(-8, 8), method="bounded").x)
    return K[np.ix_(test_idx, train_idx)] @ np.linalg.solve(Ktr + np.eye(n) / lam, ytr)


def blups(d):
    mf = smf.mixedlm("value ~ C(yr_rep)", d, groups=d["g"]).fit(reml=True); gm = d["value"].mean()
    return pd.Series({g: gm + float(v.iloc[0]) for g, v in mf.random_effects.items()})


def line_means(plots):
    df = pd.DataFrame({t: blups(plots[plots["trait"] == t]) for t in TR3})
    df["oil_yield"] = df["seed_yield"] * df["oil_content"] / 100
    df["father"] = [g.split("_", 1)[1] for g in df.index]
    return df.groupby("father").mean(numeric_only=True).reindex(fl)


def lolo(lm):
    out = {}
    for t in TR4:
        y = lm[t].to_numpy(float)
        pr = np.array([gblup_cv(y, K, [j for j in range(len(fl)) if j != k], [k])[0] for k in range(len(fl))])
        out[t] = np.corrcoef(pr, y)[0, 1]
    return out


rows = []
obs = blup[blup["father"].isin(fl)].groupby("father")[TR4].mean().reindex(fl)
ref = line_means(P)
assert np.allclose(ref[TR4].to_numpy(), obs[TR4].to_numpy(), atol=1e-6), "full-data BLUPs do not reproduce the deposited table"
rows.append(dict(analysis="main (BLUPs from all plots)", **{t: round(v, 3) for t, v in lolo(ref).items()}))

preds = {t: np.empty(len(fl)) for t in TR4}
for k, f in enumerate(fl):
    lm = line_means(P[P["father"] != f])
    tr_idx = [j for j in range(len(fl)) if j != k]
    for t in TR4:
        y = lm[t].to_numpy(float); y[k] = 0.0
        preds[t][k] = gblup_cv(y, K, tr_idx, [k])[0]
    print(f"  fold {k + 1}/{len(fl)} ({f})", flush=True)
rows.append(dict(analysis="BLUPs refitted without the held-out line", **{t: round(np.corrcoef(preds[t], obs[t])[0, 1], 3) for t in TR4}))

rp = en_table(pd.read_csv(OUT / "removed_plots.csv"))
rp = rp[rp["trait"].isin(TR3)].copy()
rp["g"] = rp["mother"] + "_" + rp["father"]; rp["yr_rep"] = rp["year"].astype(str) + "_" + rp["rep"].astype(str)
full = pd.concat([P, rp[P.columns]], ignore_index=True)
alt = line_means(full)
rows.append(dict(analysis=f"removed plot values put back (n = {len(rp)})", **{t: round(v, 3) for t, v in lolo(alt).items()}))

res = pd.DataFrame(rows)
res.to_csv(OUT / "prediction_gca_sensitivity.csv", index=False)
print("\nPredictive ability of paternal GCA (r), leave-one-line-out, 52 oilseed lines:")
print(res.to_string(index=False))
