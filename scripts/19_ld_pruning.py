# -*- coding: utf-8 -*-
"""Sensitivity of paternal GCA prediction to pruning the markers for linkage disequilibrium (Sections 2.5 and 3.2).

Pruning follows PLINK --indep-pairwise <window> <step> <r2> within chromosomes on the 54 paternal lines: of each pair
above the threshold, the marker with the higher minor-allele frequency is kept. Windows of 50 SNPs (step 5) and
1000 SNPs (step 50), r2 > 0.5 and 0.2. For each set: share of variance of the first two principal components and
leave-one-line-out GBLUP of paternal GCA on the 52 oilseed lines, as in 06_gca_prediction.py.
Output: tables/ld_pruning_sensitivity.csv.
"""

import sys
from pathlib import Path as _PathBoot
_p = _PathBoot(__file__).resolve().parent
while _p != _p.parent and not (_p / "paths.py").exists():
    _p = _p.parent
if str(_p) not in sys.path:
    sys.path.insert(0, str(_p))
from paths import OUT, en_table, en_ids
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar
import warnings; warnings.filterwarnings("ignore")

TR = ["seed_yield", "oil_content", "seed_weight_1000", "oil_yield"]
CONF = ["LI29", "LI30"]
blup = en_table(pd.read_csv(OUT / "hybrid_blup_phenotypes.csv"))
blup[["mother", "father"]] = blup["hybrid"].str.split("_", n=1, expand=True)
z = np.load(_p / "genotypes" / "merged_genotypes.npz", allow_pickle=True)
G = z["genotypes"].astype(float); G[G < 0] = np.nan


def cn(c):
    try: return int(str(c).replace("CM00", "").replace(".2", "")) - 7889
    except Exception: return -1


chrom = np.array([cn(c) for c in z["chrom"]]); pos = z["pos"].astype(int); samp = list(en_ids(z["samples"].astype(str)))
cr = np.mean(~np.isnan(G), 1); af = np.nanmean(G, 1) / 2; maf = np.minimum(af, 1 - af)
keep = (chrom >= 1) & (chrom <= 17) & (cr >= 0.9) & (maf >= 0.05)            # the 82,889-SNP set of 06_gca_prediction.py
M = G[keep]; M = np.where(np.isnan(M), np.nanmean(M, 1, keepdims=True), M)
ch, ps, mf = chrom[keep], pos[keep], maf[keep]
fathers = sorted(blup["father"].unique()); idx = {s: i for i, s in enumerate(samp)}
Mf = M[:, [idx[f] for f in fathers]]
assert Mf.shape == (82889, 54), Mf.shape


def ld_prune(X, thr, win, step):
    kp = np.ones(X.shape[0], bool)
    Xs = X - X.mean(1, keepdims=True); sd = np.sqrt((Xs ** 2).sum(1)); sd[sd == 0] = np.inf
    Xn = Xs / sd[:, None]
    for c in range(1, 18):
        ii = np.where(ch == c)[0]; ii = ii[np.argsort(ps[ii], kind="stable")]
        for s in range(0, len(ii), step):
            w = np.array([j for j in ii[s:s + win] if kp[j]])
            if len(w) < 2: continue
            A, B = np.where(np.triu((Xn[w] @ Xn[w].T) ** 2 > thr, 1))
            for a, b in zip(A, B):
                ja, jb = w[a], w[b]
                if kp[ja] and kp[jb]:
                    kp[jb if mf[ja] >= mf[jb] else ja] = False
    return kp


def kmat(X):
    p = X.mean(1) / 2; Zc = X - 2 * p[:, None]
    return (Zc.T @ Zc) / (2 * np.sum(p * (1 - p))) + np.eye(X.shape[1]) * 1e-6


def gblup(y, K, tr, te):
    ytr = y[tr] - y[tr].mean(); Ktr = K[np.ix_(tr, tr)]; n = len(tr)
    def negll(l):
        V = np.exp(l) * Ktr + np.eye(n); s, ld = np.linalg.slogdet(V); a = np.linalg.solve(V, ytr)
        return 0.5 * (ld + n * np.log(ytr @ a))
    lam = np.exp(minimize_scalar(negll, bounds=(-25, 10), method="bounded").x)
    return K[np.ix_(te, tr)] @ np.linalg.solve(Ktr + np.eye(n) / lam, ytr)


rows = []
for win, step, thr in ((None, None, None), (50, 5, 0.5), (50, 5, 0.2), (1000, 50, 0.5), (1000, 50, 0.2)):
    kp = np.ones(Mf.shape[0], bool) if win is None else ld_prune(Mf, thr, win, step)
    X = Mf[kp]; K = kmat(X)
    S = (X - X.mean(1, keepdims=True)) / np.where(X.std(1, keepdims=True) == 0, 1, X.std(1, keepdims=True))
    ev = np.linalg.svd(S, compute_uv=False) ** 2; ev = ev / ev.sum()
    row = dict(window_snps=win, step=step, r2_threshold=thr, snps=int(kp.sum()), pc1_pct=100 * ev[0], pc2_pct=100 * ev[1])
    for tr in TR:
        fm = blup.dropna(subset=[tr]).groupby("father")[tr].mean()
        fl = [f for f in fathers if f in fm.index and f not in CONF]
        y = fm.reindex(fl).to_numpy(float); ii = [fathers.index(f) for f in fl]; Ks = K[np.ix_(ii, ii)]
        pr = np.array([gblup(y, Ks, [j for j in range(len(y)) if j != k], [k])[0] for k in range(len(y))])
        row[f"r_{tr}"] = float(np.corrcoef(pr, y)[0, 1])
    rows.append(row)
    print("  " + ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
out = pd.DataFrame(rows)
assert [round(out.loc[0, f"r_{t}"], 3) for t in TR] == [0.472, 0.404, 0.655, 0.480], out.loc[0]
out.to_csv(OUT / "ld_pruning_sensitivity.csv", index=False)
print("\nsaved tables/ld_pruning_sensitivity.csv")
