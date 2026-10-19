"""Shared post-processing for the GEMMA association scans."""
from __future__ import annotations

import tempfile

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

from .external import PlinkPanel, prepare_panel, run_gemma
from .genotypes import filter_panel, standardized_pca, vanraden_kinship

CHI2_MEDIAN = stats.chi2.ppf(0.5, 1)


def genomic_inflation(pvalues) -> float:
    """Genomic-control lambda from a vector of p-values."""
    p = np.asarray(pvalues, dtype=float)
    p = p[np.isfinite(p)]
    if p.size == 0:
        return float("nan")
    return float(np.median(stats.chi2.isf(p, 1)) / CHI2_MEDIAN)


def scan_summary(assoc: pd.DataFrame, n_markers: int) -> dict:
    """The columns reported for one model/trait/panel combination."""
    p = assoc["p"].to_numpy(float)
    valid = np.isfinite(p)
    if not valid.any():
        return {
            "n_SNP": int(n_markers), "lambda": float("nan"), "h2_SNP": np.nan,
            "n_p<5e-8": 0, "n_bonf": 0, "min_p": np.nan, "top": None, "top_p": np.nan,
        }
    index = int(np.nanargmin(np.where(valid, p, np.inf)))
    row = assoc.iloc[index]
    return {
        "n_SNP": int(n_markers),
        "lambda": genomic_inflation(p),
        "h2_SNP": np.nan,  # filled by the caller from GEMMA's pve
        "n_p<5e-8": int(np.sum(p[valid] < 5e-8)),
        "n_bonf": int(np.sum(p[valid] < 0.05 / n_markers)),
        "min_p": float(p[index]),
        "top": f"chr{int(row['chr'])}:{int(row['pos'])}",
        "top_p": float(p[index]),
    }


def top_hits(assoc: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Top ``n`` markers by p-value; no loci are hard-coded anywhere."""
    columns = ["chr", "pos", "p", "beta", "marker_index"]
    if assoc.empty:
        return pd.DataFrame(columns=columns)
    return (
        assoc.loc[np.isfinite(assoc["p"])]
        .nsmallest(n, "p")[columns]
        .reset_index(drop=True)
    )


def bootstrap_minima(
    plink_panel: PlinkPanel,
    y: np.ndarray,
    covariates: np.ndarray | None,
    pve: float | None,
    nrep: int,
    *,
    seed: int = 777,
    jobs: int = 1,
    label_prefix: str = "boot",
) -> np.ndarray:
    """Parametric-bootstrap null distribution of the genome-wide minimum p.

    The null covariance uses GEMMA's own kinship matrix and its estimated PVE:
    ``s2g = pve * var(y)``, ``s2e = (1 - pve) * var(y)``.  Each replicate is a
    fresh GEMMA scan on a simulated phenotype; replicates are independent and
    can run in threads.
    """
    mask = np.isfinite(y)
    y_obs = np.asarray(y, float)[mask]
    n_obs = len(y_obs)
    n = len(y)
    kinship = np.loadtxt(plink_panel.kinship)
    kinship = kinship[np.ix_(mask, mask)]
    variance = float(np.var(y_obs))
    prop = 0.5 if pve is None or not np.isfinite(pve) else float(pve)
    prop = min(max(prop, 1e-6), 1 - 1e-6)
    a = prop * variance * kinship + (1.0 - prop) * variance * np.eye(n_obs)
    a = a + np.eye(n_obs) * 1e-9
    chol = np.linalg.cholesky(a)
    x = np.ones((n_obs, 1))
    if covariates is not None:
        x = np.column_stack([x, np.asarray(covariates)[mask]])
    a_inv_x = np.linalg.solve(a, x)
    beta = np.linalg.solve(x.T @ a_inv_x, a_inv_x.T @ y_obs)
    mu = x @ beta

    seeds = np.random.SeedSequence(seed).spawn(nrep)

    def one(index: int, child: int) -> float:
        rng = np.random.default_rng(child)
        simulated = mu + chol @ rng.standard_normal(n_obs)
        full = np.full(n, np.nan)
        full[mask] = simulated
        label = f"{label_prefix}_{index}"
        result = run_gemma(plink_panel, full, covariates=covariates, label=label)
        value = float(np.nanmin(result["assoc"]["p"]))
        for suffix in (".pheno", ".cov", ".assoc.txt", ".log"):
            (plink_panel.workdir / f"{label}{suffix}").unlink(missing_ok=True)
        return value

    args = [(i, int(s.generate_state(1)[0])) for i, s in enumerate(seeds)]
    if jobs in (0, 1):
        return np.array([one(*arg) for arg in args])
    return np.array(Parallel(n_jobs=jobs, backend="threading")(
        delayed(one)(*arg) for arg in args
    ))


def confectionery_scan(
    data,
    means: pd.DataFrame,
    traits: list[str],
    line_sets,
    *,
    with_pcs: bool = False,
    nboot: int = 200,
    jobs: int = 1,
) -> pd.DataFrame:
    """Scan with and without the two confectionery lines, with a bootstrap threshold.

    ``line_sets`` is a sequence of ``(label, lines)``.  The empirical 5%
    threshold comes from a parametric bootstrap under GEMMA's fitted null model.
    """
    rows = []
    with tempfile.TemporaryDirectory(prefix="sunflower-conf-") as tmp:
        for tag, lines in line_sets:
            panel = filter_panel(data, lines, call_rate=0.9, maf=0.05)
            scores, _ = standardized_pca(panel.dosage, 3)
            prepared = prepare_panel(
                panel, tmp, prefix=f"conf_{tag}",
                kinship=vanraden_kinship(panel.dosage),
            )
            for trait in traits:
                y = means[trait].reindex(lines).to_numpy(float)
                covariates = scores if with_pcs else None
                result = run_gemma(
                    prepared, y, covariates=covariates,
                    label=f"obs_{tag}_{trait}",
                )
                minima = bootstrap_minima(
                    prepared, y, covariates, result["pve"], nboot,
                    seed=777, jobs=jobs,
                    label_prefix=f"b{tag[:2]}_{trait[:6]}",
                )
                summary = scan_summary(result["assoc"], panel.n_markers)
                threshold = float(np.quantile(minima, 0.05))
                below = int(np.sum(result["assoc"]["p"].to_numpy(float) < threshold))
                rows.append({
                    "callset": tag, "trait": trait, "n_SNP": panel.n_markers,
                    "λ": summary["lambda"], "n_p<5e-8": summary["n_p<5e-8"],
                    "emp_threshold_5pct": threshold, "n_below_emp": below,
                    "top": summary["top"], "top_p": summary["top_p"],
                })
                print(
                    f"  {trait:26s} lambda={summary['lambda']:.2f}  "
                    f"p<5e-8: {summary['n_p<5e-8']}  emp.thr={threshold:.1e} "
                    f"-> significant {below}  top {summary['top']} p={summary['top_p']:.1e}",
                    flush=True,
                )
    return pd.DataFrame(rows)


def loci_table(assoc: pd.DataFrame, loci: pd.DataFrame, *, default_window: int = 0) -> pd.DataFrame:
    """p-values at user-supplied loci.

    ``loci`` must have ``locus, chr, pos`` and may have ``window``.  The
    original script hard-coded five loci and wrote them as if they were
    results; now the list is an explicit input and is never inferred.
    """
    rows = []
    for _, locus in loci.iterrows():
        window = int(locus.get("window", default_window))
        hit = assoc[
            (assoc["chr"] == int(locus["chr"]))
            & (assoc["pos"].between(int(locus["pos"]) - window, int(locus["pos"]) + window))
        ]
        rows.append({
            "locus": locus["locus"],
            "chr": int(locus["chr"]),
            "pos": int(locus["pos"]),
            "window": window,
            "present": not hit.empty,
            "markers_in_window": len(hit),
            "p": float(hit["p"].min()) if not hit.empty else np.nan,
        })
    return pd.DataFrame(rows)
