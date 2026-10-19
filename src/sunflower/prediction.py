"""Shared genomic-prediction helpers.

The direct GBLUP formulas live here once instead of being copied into scripts
06, 07, 08, 19 and 20.  They are cross-checked against R/rrBLUP in
``tests/test_gblup_matches_rrblup.py``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import optimize

from .genotypes import hybrid_dosage

CONF = ("LI29", "LI30")
TESTER_MAP = {"VK101A": "VK101", "VA761A": "VA761", "VK934A": "VK934"}
TESTERS = ("VK101", "VA761", "VK934")


def reml_variance(y: np.ndarray, K: np.ndarray, X: np.ndarray):
    """REML estimates of (s2g, s2e) for y = X b + u + e, u ~ N(0, s2g K)."""
    ev, U = np.linalg.eigh(K)
    ev = np.maximum(ev, 1e-9)
    yr = U.T @ y
    xr = U.T @ X

    def negll(theta):
        s2g, s2e = np.exp(theta)
        d = s2g * ev + s2e
        if (d <= 0).any():
            return 1e10
        w = 1.0 / d
        xtwx = xr.T @ (xr * w[:, None])
        try:
            inv = np.linalg.inv(xtwx)
        except np.linalg.LinAlgError:
            return 1e10
        beta = inv @ (xr.T @ (yr * w))
        resid = yr - xr @ beta
        _, logdet = np.linalg.slogdet(xtwx)
        return 0.5 * (np.log(d).sum() + (resid**2 * w).sum() + logdet)

    variance = float(np.var(y))
    result = optimize.minimize(
        negll, np.log([0.5 * variance, 0.5 * variance]),
        method="L-BFGS-B", bounds=[(np.log(1e-8), np.log(1e8))] * 2,
    )
    if not np.isfinite(result.x).all():
        s2g = s2e = variance / 2.0
    else:
        floor = max(variance, 1e-12) * 1e-8
        s2g = max(float(np.exp(result.x[0])), floor)
        s2e = max(float(np.exp(result.x[1])), floor)
    return s2g, s2e, ev, U


def gblup_predict(ytr: np.ndarray, Xtr: np.ndarray, Ktr: np.ndarray,
                  Kcross: np.ndarray, Xte: np.ndarray) -> np.ndarray:
    """BLUP prediction for new individuals with fixed effects."""
    s2g, s2e, ev, U = reml_variance(ytr, Ktr, Xtr)
    vi = U @ np.diag(1.0 / (s2g * ev + s2e)) @ U.T
    beta = np.linalg.lstsq(Xtr.T @ vi @ Xtr, Xtr.T @ vi @ ytr, rcond=None)[0]
    return Xte @ beta + s2g * (Kcross @ (vi @ (ytr - Xtr @ beta)))


def gblup_cv(y: np.ndarray, K: np.ndarray, train_idx, test_idx) -> np.ndarray:
    """Leave-fold-out GBLUP for paternal-line means.

    The variance ratio is profiled on the training part only (as in the
    original script 06); K is the VanRaden relationship matrix.
    """
    ytr = y[train_idx]
    ytr = ytr - ytr.mean()
    ktr = K[np.ix_(train_idx, train_idx)]
    n = len(ytr)

    def negll(loglam):
        lam = np.exp(loglam)
        v = ktr + np.eye(n) / lam
        _, logdet = np.linalg.slogdet(v)
        alpha = np.linalg.solve(v, ytr)
        return 0.5 * (logdet + n * np.log(ytr @ alpha))

    lam = np.exp(optimize.minimize_scalar(negll, bounds=(-8, 8), method="bounded").x)
    alpha = np.linalg.solve(ktr + np.eye(n) / lam, ytr)
    return K[np.ix_(test_idx, train_idx)] @ alpha


def hybrid_kinship(dosage: np.ndarray, samples: tuple[str, ...],
                   mothers: list[str], fathers: list[str]) -> np.ndarray:
    """VanRaden kinship of hybrid mid-parent genotypes."""
    hg = hybrid_dosage(dosage, samples, mothers, fathers)
    af = hg.mean(axis=0) / 2.0
    w = hg - 2.0 * af
    denom = 2.0 * float(np.sum(af * (1.0 - af)))
    return (w @ w.T) / denom + np.eye(len(hg)) * 1e-4


def xmat(frame: pd.DataFrame, mother_col: str = "mother") -> np.ndarray:
    """Intercept plus tester indicators (baseline VK101)."""
    mother = frame[mother_col].to_numpy()
    return np.column_stack([
        np.ones(len(frame)),
        (mother == "VA761").astype(float),
        (mother == "VK934").astype(float),
    ])


def build_hybrids(blup: pd.DataFrame, samples, conf=CONF) -> pd.DataFrame:
    """Normalise the hybrid BLUP table to mother/father names used by genotypes."""
    out = blup.copy()
    out[["mother_long", "father"]] = out["hybrid"].str.split("_", n=1, expand=True)
    out["mother"] = out["mother_long"].map(TESTER_MAP)
    out = out[out["mother"].isin(set(samples)) & out["father"].isin(set(samples))]
    if conf:
        out = out[~out["father"].isin(set(conf))]
    return out.reset_index(drop=True)


def cv_predictions(frame: pd.DataFrame, trait: str, K: np.ndarray,
                   nfold: int = 5, mother_col: str = "mother") -> pd.DataFrame:
    """Five-fold (or leave-one-line-out) predictions for individual hybrids.

    Folds are assigned by father line so all hybrids of a father stay together.
    """
    d = frame.dropna(subset=[trait]).reset_index(drop=True)
    y = d[trait].to_numpy(float)
    x = xmat(d, mother_col)
    fathers = sorted(d["father"].unique())
    fold_of = {father: i % nfold for i, father in enumerate(fathers)}
    fold = d["father"].map(fold_of).to_numpy()
    predicted = np.full(len(d), np.nan)
    for k in range(nfold):
        test = np.where(fold == k)[0]
        train = np.where(fold != k)[0]
        if len(test) == 0 or len(train) < 10:
            continue
        predicted[test] = gblup_predict(
            y[train], x[train], K[np.ix_(train, train)],
            K[np.ix_(test, train)], x[test],
        )
    return d.assign(y=y, yp=predicted).dropna(subset=["yp"])


def hybrid_blups(plot: pd.DataFrame, traits: list[str], return_stats: bool = False):
    """BLUP phenotypes for hybrids (year-plot fixed, combination random)."""
    out = {}
    stats = {}
    for trait in traits:
        d = plot.loc[plot["trait"] == trait].copy()
        d["g"] = d["mother"] + "_" + d["father"]
        d["yr_rep"] = d["year"].astype(str) + "_" + d["rep"].astype(str)
        fit = smf.mixedlm("value ~ C(yr_rep)", d, groups=d["g"]).fit(reml=True)
        s2g = float(fit.cov_re.iloc[0, 0])
        s2e = float(fit.scale)
        r = len(d) / d["g"].nunique()
        stats[trait] = {
            "n": len(d), "combinations": d["g"].nunique(),
            "s2g": s2g, "s2e": s2e,
            "repeatability": s2g / (s2g + s2e),
            "H2": s2g / (s2g + s2e / r),
        }
        grand = float(d["value"].mean())
        out[trait] = {
            group: grand + float(effect.iloc[0])
            for group, effect in fit.random_effects.items()
        }
    frame = pd.DataFrame(out).rename_axis("hybrid").reset_index()
    return (frame, stats) if return_stats else frame


def paternal_means(blup: pd.DataFrame, traits: list[str]) -> pd.DataFrame:
    return blup.groupby("father")[traits].mean()
