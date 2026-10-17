"""Genomic prediction accuracy for individual hybrids using a leave-one-father-out
(LOFO) cross-validation scheme, the same scheme used for genomic prediction of a
father's GCA.

For each trait, hybrids are split into folds where each father (pollinator line) is
its own fold, and GBLUP is fit on hybrid genotypes with a fixed-tester covariate;
confectionery lines are excluded for oil content and for seed weight per 1000 seeds.
As a control, the identical prediction code is also run with a fixed 5-fold split
over fathers, and the resulting overall correlation is asserted to match the value
already reported for the standard 5-fold cross-validation.

Inputs:
    tables/hybrid_blup_phenotypes.csv   - per-hybrid BLUP phenotypes
    genotypes/merged_genotypes.npz      - merged marker genotypes
    tables/prediction_accuracy_5fold.csv - reference 5-fold accuracy (control values)

Outputs:
    tables/prediction_leave_one_line_out.csv
    tables/prediction_leave_one_line_out.txt
"""
import sys
from pathlib import Path as _PathBoot
_p = _PathBoot(__file__).resolve().parent
while _p != _p.parent and not (_p / "paths.py").exists():
    _p = _p.parent
if str(_p) not in sys.path:
    sys.path.insert(0, str(_p))
from paths import OUT, DEPOSIT, HERE as REPO_ROOT, use_models, en_table, en_ids, en_id
use_models()          # makes scripts/models importable: from baker_v4 import reml_fit, moments

import warnings
import numpy as np, pandas as pd
from scipy import stats, optimize
warnings.filterwarnings("ignore")

CONF = {"LI29", "LI30"}
LTV = {"VK101A": "VK101", "VA761A": "VA761", "VK934A": "VK934"}
CHRMAP = {f"CM00{7889+i}.2": i for i in range(1, 18)}


# ---- unchanged from the phase-14 recomputation code ----
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
bl[["mother_long", "father"]] = bl["hybrid"].str.split("_", expand=True)
bl["mother"] = bl["mother_long"].map(LTV)
bl = bl[bl["mother"].isin(set(samples)) & bl["father"].isin(set(samples))].reset_index(drop=True)


def build(trait, drop_conf):
    d = bl.dropna(subset=[trait]).copy()
    if drop_conf: d = d[~d["father"].isin(CONF)]
    return d.reset_index(drop=True)


def hg_of(d):
    HG = np.zeros((len(d), n_snp))
    for i, row in d.iterrows(): HG[i] = (gt[:, s2c[row["mother"]]] + gt[:, s2c[row["father"]]]) / 2
    return HG


def Xmat(d): return np.column_stack([np.ones(len(d)), (d["mother"] == "VA761").astype(float), (d["mother"] == "VK934").astype(float)])


def gpred(ytr, Xtr, Ktr, Kc, Xte):
    s2g, s2e, ev, U = vc(ytr, Ktr, Xtr); Vi = U @ np.diag(1 / (s2g * ev + s2e)) @ U.T
    beta = np.linalg.inv(Xtr.T @ Vi @ Xtr) @ (Xtr.T @ Vi @ ytr)
    return Xte @ beta + s2g * (Kc @ (Vi @ (ytr - Xtr @ beta))), s2g, s2e
# ---- end of carried-over code ----


def cv1_predictions(trait, dc, nfold=5):
    """Body of cv1 unchanged; returns the predictions themselves, not just the correlation."""
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
    d5 = cv1_predictions(trait, dc, nfold=5)                       # control: five-fold scheme
    r5 = np.corrcoef(d5["y"], d5["yp"])[0, 1]
    assert round(r5, 3) == acc.loc[trait, "GS_accuracy_CV1"], (trait, r5, acc.loc[trait, "GS_accuracy_CV1"])
    nf = build(trait, dc)["father"].nunique()
    d = cv1_predictions(trait, dc, nfold=nf)                        # LOFO: each line is its own fold
    assert d.groupby("father").ngroups == nf
    r_all = np.corrcoef(d["y"], d["yp"])[0, 1]
    within = {}
    for m, g in d.groupby("mother"):
        r, p = stats.pearsonr(g["y"], g["yp"]); within[m] = (r, p, len(g))
    zbar = np.mean([np.arctanh(v[0]) for v in within.values()])
    row = dict(trait=trait, n_hybrids=len(d), n_fathers=nf, r_all_lofo=r_all, r_within_mean_lofo=np.tanh(zbar),
               r_all_5fold_control=r5)
    for m in ("VK101", "VA761", "VK934"):
        row[f"r_{m}"], row[f"p_{m}"], row[f"n_{m}"] = within[m]
    rows.append(row)
    txt.append(f"{trait:12s} n={len(d)} fathers={nf}  LOFO: overall r={r_all:.3f}, within tester "
               + ", ".join(f"{m} {within[m][0]:.3f}" for m in ("VK101", "VA761", "VK934"))
               + f", Fisher average {np.tanh(zbar):.3f}  | control 5-fold r={r5:.3f} (matched prediction_accuracy_5fold)")

pd.DataFrame(rows).to_csv(OUT / "prediction_leave_one_line_out.csv", index=False, encoding="utf-8-sig")
(OUT / "prediction_leave_one_line_out.txt").write_text("\n".join(txt), encoding="utf-8")
print("\n".join(txt))