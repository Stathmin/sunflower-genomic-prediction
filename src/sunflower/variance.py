"""Variance components and Baker's ratio via R/sommer.

The 361-line hand-written REML engine from the original deposit is gone.  The
models are fitted by ``sommer::mmer`` (a well-tested REML implementation); this
module only prepares the data, calls the R helper and derives the reported
quantities (Baker's ratio, H², shares and profile intervals).
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.linalg as sla
from joblib import Parallel, delayed

from .external import run_rscript

R_HELPER = Path(__file__).parent / "r" / "variance.R"
LABELS = ["line", "tester_x_line", "year_x_line", "year_x_tester_x_line", "error"]
CRIT = 3.841458820694124  # chi2.ppf(0.95, 1)


def _run_r(mode: str, frame: pd.DataFrame, extra: list[str] | None = None) -> pd.DataFrame:
    with tempfile.TemporaryDirectory(prefix="sunflower-var-") as tmp:
        source = Path(tmp) / "input.csv"
        target = Path(tmp) / "output.csv"
        frame.to_csv(source, index=False)
        run_rscript(R_HELPER, [mode, source, target, *(extra or [])], cwd=tmp)
        return pd.read_csv(target)


def _parallel(func, items, jobs: int):
    if jobs in (0, 1) or len(items) <= 1:
        return [func(item) for item in items]
    return Parallel(n_jobs=jobs, backend="threading")(delayed(func)(item) for item in items)


def hybrid_components(plot: pd.DataFrame, traits: list[str], jobs: int = 1) -> pd.DataFrame:
    """Fit the hybrid variance-component model for each trait.

    ``plot`` needs columns ``trait, year, rep, mother, line, value``.
    """
    columns = ["trait", "year", "rep", "mother", "line", "value"]

    def one(trait: str) -> pd.DataFrame:
        subset = plot.loc[plot["trait"] == trait, columns]
        return _run_r("hybrid", subset)

    return pd.concat(_parallel(one, traits, jobs), ignore_index=True)


def line_heritability(lines: pd.DataFrame, traits: list[str], jobs: int = 1) -> pd.DataFrame:
    """Fit the paternal-line model for each trait.

    ``lines`` needs columns ``trait, year, genotype, value``.
    """
    columns = ["trait", "year", "genotype", "value"]

    def one(trait: str) -> pd.DataFrame:
        subset = lines.loc[lines["trait"] == trait, columns]
        return _run_r("line", subset)

    return pd.concat(_parallel(one, traits, jobs), ignore_index=True)


def profile_trait(plot: pd.DataFrame, trait: str, grid) -> pd.DataFrame:
    columns = ["trait", "year", "rep", "mother", "line", "value"]
    subset = plot.loc[plot["trait"] == trait, columns]
    text = ",".join(f"{float(b):.12g}" for b in grid)
    return _run_r("profile", subset, extra=[text])


def baker_ratio(frame: pd.DataFrame) -> pd.Series:
    denom = 2.0 * frame["s2_line"] + frame["s2_ml"]
    return 2.0 * frame["s2_line"] / denom


def baker_stats(row: pd.Series) -> dict:
    line, ml = float(row["s2_line"]), float(row["s2_ml"])
    denom = 2.0 * line + ml
    baker = 2.0 * line / denom
    g_line = 2.0 * ml / denom**2
    g_ml = -2.0 * line / denom**2
    var = (
        (g_line**2) * float(row["se_line"]) ** 2
        + (g_ml**2) * float(row["se_ml"]) ** 2
        + 2.0 * g_line * g_ml * float(row["cov_line_ml"])
    )
    return {"baker": baker, "se_baker": float(np.sqrt(max(var, 0.0)))}


def derived_components(row: pd.Series, ny: float, r: float) -> dict:
    s = row
    gca = float(s["s2_line"]) + float(s["s2_ml"])
    gxy = float(s["s2_yl"]) + float(s["s2_yml"])
    error = float(s["s2_e"])
    total = gca + gxy + error
    return {
        "gca_share": gca / total,
        "plot_repeatability": gca / total,
        "H2": gca / (gca + gxy / ny + error / (ny * r)),
    }


def shares_and_se(row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    values = np.array([
        float(row["s2_line"]), float(row["s2_ml"]), float(row["s2_yl"]),
        float(row["s2_yml"]), float(row["s2_e"]),
    ])
    total = values.sum()
    shares = values / total
    cov = np.array([
        [float(row.get(f"cov{i}{j}", np.nan)) for j in range(1, 6)]
        for i in range(1, 6)
    ])
    jac = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            jac[i, j] = ((1.0 if i == j else 0.0) * total - values[i]) / total**2
    variance = np.diag(jac @ cov @ jac.T)
    return shares, np.sqrt(np.clip(variance, 0.0, None))


def profile_interval(
    plot: pd.DataFrame,
    trait: str,
    *,
    step: float = 0.05,
    crit: float = CRIT,
    bisect_steps: int = 12,
) -> dict:
    """Profile-likelihood interval for Baker's ratio.

    The constrained fits are ``sommer`` models with the two variance components
    combined through ``overlay()``; ``AIC`` differences between these equally
    parameterised fits are twice the log-likelihood differences.
    """
    grid = np.round(np.arange(0.0, 1.0 + 1e-12, step), 6)
    coarse = profile_trait(plot, trait, grid)
    dev = coarse["aic"] - coarse["aic"].min()

    def deviance(b: float) -> float:
        one = profile_trait(plot, trait, [b])
        return float(one["aic"].iloc[0] - coarse["aic"].min())

    inside = coarse.loc[dev <= crit, "b"]
    low, high = float(inside.min()), float(inside.max())
    if low > 0:
        outer, inner = max(0.0, low - step), low
        for _ in range(bisect_steps):
            mid = 0.5 * (outer + inner)
            if deviance(mid) <= crit:
                inner = mid
            else:
                outer = mid
        low = inner
    if high < 1:
        inner, outer = high, min(1.0, high + step)
        for _ in range(bisect_steps):
            mid = 0.5 * (inner + outer)
            if deviance(mid) <= crit:
                inner = mid
            else:
                outer = mid
        high = inner
    profile = {f"{float(b):.4f}": float(v) for b, v in zip(coarse["b"], dev)}
    return {"ci_low": low, "ci_high": high, "profile": profile}


def profile_all(plot: pd.DataFrame, traits: list[str], jobs: int = 1) -> dict[str, dict]:
    def one(trait: str) -> tuple[str, dict]:
        return trait, profile_interval(plot, trait)

    return dict(_parallel(one, traits, jobs))


def _dummies(labels) -> np.ndarray:
    levels = sorted(set(labels))
    index = {label: i for i, label in enumerate(levels)}
    out = np.zeros((len(labels), len(levels)))
    for i, label in enumerate(labels):
        out[i, index[label]] = 1.0
    return out


def _inter(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.einsum("ij,ik->ijk", a, b).reshape(a.shape[0], -1)


def _seq_anova(y: np.ndarray, terms, tol: float = 1e-9) -> dict:
    n = len(y)
    q = np.ones((n, 1)) / np.sqrt(n)
    rows = {}
    for name, matrix in terms:
        resid = matrix - q @ (q.T @ matrix)
        qn, rn, _ = sla.qr(resid, mode="economic", pivoting=True)
        diagonal = np.abs(np.diag(rn))
        rank = int(np.sum(diagonal > tol * max(diagonal[0], np.linalg.norm(matrix))))
        if rank > 0:
            basis = qn[:, :rank]
            basis = basis - q @ (q.T @ basis)
            basis, _ = sla.qr(basis, mode="economic")
            rows[name] = {"SS": float(np.sum((basis.T @ y) ** 2)), "df": rank}
            q = np.hstack([q, basis])
        else:
            rows[name] = {"SS": 0.0, "df": 0}
    rows["error"] = {
        "SS": float(y @ y) - float(np.sum((q.T @ y) ** 2)),
        "df": n - q.shape[1],
    }
    for value in rows.values():
        value["MS"] = value["SS"] / value["df"] if value["df"] > 0 else np.nan
    return rows


def moments_baker(plot: pd.DataFrame, trait: str) -> float:
    """Method-of-moments Baker's ratio, kept as an independent check."""
    d = plot.loc[plot["trait"] == trait]
    y = d["value"].to_numpy(float)
    year = _dummies(d["year"].astype(str).tolist())
    rep = _dummies((d["year"].astype(str) + "_" + d["rep"].astype(str)).tolist())
    mother = _dummies(d["mother"].astype(str).tolist())
    line = _dummies(d["line"].astype(str).tolist())
    ml, ym, yl = _inter(mother, line), _inter(year, mother), _inter(year, line)
    yml = _inter(ym, line)
    anova = _seq_anova(y, [
        ("year", year), ("rep_in_year", rep), ("mother", mother),
        ("year_x_mother", ym), ("line", line), ("mother_x_line", ml),
        ("year_x_line", yl), ("year_x_mother_x_line", yml),
    ])
    r = float(d.groupby(["year", "mother", "line"]).size().mean())
    ny = float(d["year"].nunique())
    nm = float(d["mother"].nunique())
    ms = {key: value["MS"] for key, value in anova.items()}
    yml_ms, yl_ms, ml_ms, l_ms = (
        ms["year_x_mother_x_line"], ms["year_x_line"],
        ms["mother_x_line"], ms["line"],
    )
    s2_ml = max(0.0, (ml_ms - yml_ms) / (r * ny))
    s2_l = max(0.0, (l_ms - ml_ms - yl_ms + yml_ms) / (r * ny * nm))
    return float(2.0 * s2_l / (2.0 * s2_l + s2_ml))
