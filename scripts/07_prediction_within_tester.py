import sys
from pathlib import Path as _PathBoot
_p = _PathBoot(__file__).resolve().parent
while _p != _p.parent and not (_p / "paths.py").exists():
    _p = _p.parent
if str(_p) not in sys.path:
    sys.path.insert(0, str(_p))
from paths import OUT, DEPOSIT, HERE as REPO_ROOT, use_models, en_table, en_ids, en_id
use_models()          # makes scripts/models importable: from baker_v4 import reml_fit, moments

"""Genomic-prediction accuracy for hybrids, computed within each tester line.

Purpose. The pooled hybrid-level correlations between observed and
GBLUP-predicted trait values (as reported in prediction_accuracy_5fold.csv)
include the difference between the three tester lines, which is easy to
predict because tester is a fixed effect in the model. What a breeder
actually needs to know is how well the prediction ranks father lines
*within* a given tester. This script reuses the exact same GBLUP
prediction pipeline (fixed tester effect, hybrid genotypes as the average
of parental genotypes, 5-fold cross-validation with folds assigned by
father line so that all hybrids of a father stay in one fold,
confectionery lines excluded for oil content and 1000-seed weight) and
additionally reports the observed-vs-predicted correlation computed
separately within each tester, along with their Fisher-averaged mean.

The pooled correlation is checked against the previously computed
prediction_accuracy_5fold.csv and the script raises if the two disagree.

Inputs:
    tables/hybrid_blup_phenotypes.csv
    genotypes/merged_genotypes.npz
    tables/prediction_accuracy_5fold.csv (consistency check only)

Outputs:
    tables/prediction_within_tester_5fold.csv
    tables/prediction_within_tester_5fold.txt
"""
import warnings
import numpy as np, pandas as pd
from scipy import stats, optimize
warnings.filterwarnings("ignore")

CONF = {"LI29", "LI30"}
LTV = {"VK101A": "VK101", "VA761A": "VA761", "VK934A": "VK934"}
CHRMAP = {f"CM00{7889 + i}.2": i for i in range(1, 18)}


# ---- unchanged from the accuracy-computation pipeline ----
def reml(t, yr, Xr, ev):
    s2g = np.exp(t[0]); s2e = np.exp(t[1]); Dd = s2g * ev + s2e
    if (Dd <= 0).any(): return 1e10
    W = 1 / Dd; XtWX = Xr.T @ (Xr * W[:, None])
    try: inv = np.linalg.inv(XtWX)
    except Exception: return 1e10
    b = inv @ (Xr.T @ (yr * W)); r = yr - Xr @ b; _, ld = np.linalg.slogdet(XtWX)
    return 0.5 * (np.log(Dd).sum() + (r ** 2 * W).sum() + ld)


def vc(y, K, X):
    ev, U = np.linalg.eigh(K); ev = np.maximum(ev, 1e-9); yr = U.T @ y; Xr = U.T @ X; vy = np.var(y)
    res = optimize.minimize(reml, np.log([.5 * vy, .5 * vy]), args=(yr, Xr, ev), method="L-BFGS-B",
                            bounds=[(np.log(1e-8), np.log(1e8))] * 2)
    return np.exp(res.x[0]), np.exp(res.x[1]), ev, U


z = np.load(DEPOSIT / "merged_genotypes.npz", allow_pickle=True)
disc = z["genotypes"].astype(float); chrom = z["chrom"].astype(str); samples = list(z["samples"].astype(str))
nalt = np.where(disc < 0, np.nan, disc); chrn = np.array([CHRMAP.get(c, -1) for c in chrom])
cr = np.mean(~np.isnan(nalt), 1); af = np.nanmean(nalt, 1) / 2; maf = np.minimum(af, 1 - af)
keep = (chrn >= 1) & (chrn <= 17) & (cr >= 0.9) & (maf >= 0.05)
gt = nalt[keep]; gt = np.where(np.isnan(gt), np.nanmean(gt, axis=1, keepdims=True), gt)
n_snp = gt.shape[0]; s2c = {s: i for i, s in enumerate(samples)}
bl = pd.read_csv(OUT / "hybrid_blup_phenotypes.csv")
bl[["mom_long", "father"]] = bl["hybrid"].str.split("_", expand=True)
bl["mom"] = bl["mom_long"].map(LTV)
bl = bl[bl["mom"].isin(set(samples)) & bl["father"].isin(set(samples))].reset_index(drop=True)


def build(trait, drop_conf):
    d = bl.dropna(subset=[trait]).copy()
    if drop_conf: d = d[~d["father"].isin(CONF)]
    return d.reset_index(drop=True)


def hg_of(d):
    HG = np.zeros((len(d), n_snp))
    for i, row in d.iterrows(): HG[i] = (gt[:, s2c[row["mom"]]] + gt[:, s2c[row["father"]]]) / 2
    return HG


def Xmat(d): return np.column_stack([np.ones(len(d)), (d["mom"] == "VA761").astype(float), (d["mom"] == "VK934").astype(float)])


def gpred(ytr, Xtr, Ktr, Kc, Xte):
    s2g, s2e, ev, U = vc(ytr, Ktr, Xtr); Vi = U @ np.diag(1 / (s2g * ev + s2e)) @ U.T
    beta = np.linalg.inv(Xtr.T @ Vi @ Xtr) @ (Xtr.T @ Vi @ ytr)
    return Xte @ beta + s2g * (Kc @ (Vi @ (ytr - Xtr @ beta))), s2g, s2e
# ---- end of transferred code ----


def cv1_predictions(trait, dc, nfold=5):
    """Body of the cross-validation unchanged; returns the predictions
    themselves rather than only the aggregate correlation."""
    d = build(trait, dc); HG = hg_of(d); af_h = HG.mean(0) / 2; Wh = HG - 2 * af_h
    K = (Wh @ Wh.T) / (2 * np.sum(af_h * (1 - af_h))) + np.eye(len(d)) * 1e-4
    y = d[trait].values.astype(float); X = Xmat(d)
    fa = sorted(d["father"].unique()); fo = {f: i % nfold for i, f in enumerate(fa)}; fold = d["father"].map(fo).values
    yp = np.full(len(d), np.nan)
    for k in range(nfold):
        te = np.where(fold == k)[0]; tr = np.where(fold != k)[0]
        if len(te) == 0 or len(tr) < 10: continue
        pr, _, _ = gpred(y[tr], X[tr], K[np.ix_(tr, tr)], K[np.ix_(te, tr)], X[te]); yp[te] = pr
    d = d.assign(y=y, yp=yp)
    return d[~np.isnan(yp)]


acc = pd.read_csv(OUT / "prediction_accuracy_5fold.csv", comment="#").set_index("trait")
rows, txt = [], [f"SNPs: {n_snp:,}; hybrids with genotypes: {len(bl)}", ""]
for trait, dc in [("seed_weight_1000", True), ("oil_content", True), ("oil_yield", True), ("seed_yield", True)]:
    d = cv1_predictions(trait, dc)
    r_all = np.corrcoef(d["y"], d["yp"])[0, 1]
    assert round(r_all, 3) == acc.loc[trait, "GS_accuracy_CV1"], \
        ("overall correlation does not match prediction_accuracy_5fold.csv", trait, r_all, acc.loc[trait, "GS_accuracy_CV1"])
    within = {}
    for m, g in d.groupby("mom"):
        r, p = stats.pearsonr(g["y"], g["yp"]); within[m] = (r, p, len(g))
    zbar = np.mean([np.arctanh(v[0]) for v in within.values()])
    # share of variance in observed values attributable to differences between testers
    tester_share = d.groupby("mom")["y"].transform("mean").var(ddof=0) / d["y"].var(ddof=0)
    row = dict(trait=trait, n_hybrids=len(d), r_all=r_all, r_within_mean=np.tanh(zbar),
               between_tester_variance_share=tester_share)
    for m in ("VK101", "VA761", "VK934"):
        row[f"r_{m}"], row[f"p_{m}"], row[f"n_{m}"] = within[m]
    rows.append(row)
    txt.append(f"{trait:17s} n={len(d)}  overall r={r_all:.3f} (matches prediction_accuracy_5fold)  within tester: "
               + ", ".join(f"{m} {within[m][0]:.3f} (n={within[m][2]}, p={within[m][1]:.2g})" for m in ("VK101", "VA761", "VK934"))
               + f"  | Fisher-averaged mean {np.tanh(zbar):.3f} | between-tester variance share {tester_share * 100:.1f} %")

pd.DataFrame(rows).to_csv(OUT / "prediction_within_tester_5fold.csv", index=False, encoding="utf-8-sig")
(OUT / "prediction_within_tester_5fold.txt").write_text("\n".join(txt), encoding="utf-8")
print("\n".join(txt))