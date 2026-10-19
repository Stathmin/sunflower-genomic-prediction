"""Dominance-deviation scan (two kinship kernels).

This is the paper-specific model ``y ~ 1 + tester + SNP_dose + SNP_het`` with
``u_a ~ N(0, s2a Ka)`` and ``u_d ~ N(0, s2d Kd)``.  It has no off-the-shelf
equivalent, so the scan itself remains here; the VCF reader is pysam and the
null replicates run in parallel with one deterministic seed per replicate.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import optimize, stats
from threadpoolctl import threadpool_limits

from .genotypes import GenotypeData, filter_panel, hybrid_dosage, read_vcf_dosage
from .io import write_csv
from .prediction import CONF, build_hybrids

MODES = ("bootstrap", "permutation", "permutation_within")
BASE_SEED = 20260816


@dataclass
class Kernels:
    hg: np.ndarray
    ih: np.ndarray
    ka: np.ndarray
    kd: np.ndarray
    x: np.ndarray
    useful: np.ndarray
    y: np.ndarray
    mothers: np.ndarray
    n: int
    n_snp: int
    df: int


def build_kernels(panel, d: pd.DataFrame, trait: str) -> Kernels:
    y = d[trait].to_numpy(float)
    hg = hybrid_dosage(panel.dosage, panel.samples, d["mother"].tolist(), d["father"].tolist())
    ih = (np.abs(hg - 1.0) < 0.25).astype(float)
    p = hg.mean(axis=0) / 2.0
    wa = hg - 2.0 * p
    ka = (wa @ wa.T) / (2.0 * float(np.sum(p * (1.0 - p))))
    expected_het = 2.0 * p * (1.0 - p)
    wd = ih - expected_het
    kd = (wd @ wd.T) / float(np.sum(expected_het * (1.0 - expected_het)))
    x = np.column_stack([
        np.ones(len(d)),
        (d["mother"] == "VA761").astype(float),
        (d["mother"] == "VK934").astype(float),
    ])
    useful = (
        (ih.sum(axis=0) >= 5)
        & (ih.std(axis=0) > 1e-6)
        & (hg.std(axis=0) > 1e-6)
    )
    n = len(d)
    return Kernels(
        hg=hg, ih=ih, ka=ka, kd=kd, x=x, useful=useful, y=y,
        mothers=d["mother"].to_numpy(), n=n, n_snp=panel.n_markers,
        df=n - x.shape[1] - 2,
    )


def _reml_profile(h, y, kernels: Kernels) -> float:
    ha, hd = h
    if ha < 0 or hd < 0 or ha + hd > 0.999:
        return 1e10
    n = kernels.n
    a = ha * kernels.ka + hd * kernels.kd + (1.0 - ha - hd) * np.eye(n)
    try:
        chol = np.linalg.cholesky(a)
    except np.linalg.LinAlgError:
        return 1e10
    ai_x = np.linalg.solve(a, kernels.x)
    xtax = kernels.x.T @ ai_x
    try:
        beta = np.linalg.solve(xtax, ai_x.T @ y)
    except np.linalg.LinAlgError:
        return 1e10
    resid = y - kernels.x @ beta
    q = resid @ np.linalg.solve(a, resid)
    if q <= 0:
        return 1e10
    _, logdet_x = np.linalg.slogdet(xtax)
    n_coef = kernels.x.shape[1]
    return 0.5 * (
        2.0 * np.sum(np.log(np.diag(chol)))
        + logdet_x
        + (n - n_coef) * np.log(q / (n - n_coef))
    )


def fit_vc(y: np.ndarray, kernels: Kernels, starts) -> tuple[float, float]:
    best = None
    for start in starts:
        result = optimize.minimize(
            _reml_profile, np.asarray(start, float), args=(y, kernels),
            method="Nelder-Mead",
            options={"maxiter": 400, "fatol": 1e-7, "xatol": 1e-5},
        )
        if best is None or result.fun < best.fun:
            best = result
    ha, hd = np.clip(best.x, 0, None)
    if ha + hd > 0.999:
        ha, hd = ha / (ha + hd) * 0.999, hd / (ha + hd) * 0.999
    return float(ha), float(hd)


def scan(y: np.ndarray, ha: float, hd: float, kernels: Kernels):
    n = kernels.n
    a = ha * kernels.ka + hd * kernels.kd + (1.0 - ha - hd) * np.eye(n)
    eigenvalues, u = np.linalg.eigh(a)
    eigenvalues = np.maximum(eigenvalues, 1e-10)
    sw = 1.0 / np.sqrt(eigenvalues)
    xw = (u.T @ kernels.x) * sw[:, None]
    q, _ = np.linalg.qr(xw)
    yw = (u.T @ y) * sw
    yt = yw - q @ (q.T @ yw)
    gt = (u.T @ kernels.hg) * sw[:, None]
    gt -= q @ (q.T @ gt)
    ht = (u.T @ kernels.ih) * sw[:, None]
    ht -= q @ (q.T @ ht)
    gg = np.einsum("ij,ij->j", gt, gt)
    gg = np.where(gg < 1e-12, np.nan, gg)
    gh = np.einsum("ij,ij->j", gt, ht)
    gy = gt.T @ yt
    hp = ht - gt * (gh / gg)
    del gt, ht
    hh = np.einsum("ij,ij->j", hp, hp)
    hy = hp.T @ yt
    ok = kernels.useful & np.isfinite(hh) & (hh > 1e-10) & np.isfinite(gg)
    beta = hy / np.where(ok, hh, np.nan)
    sse = (yt @ yt) - (gy**2) / gg - beta**2 * hh
    sse = np.where(sse > 1e-12, sse, np.nan)
    t = beta / np.sqrt(sse / kernels.df / hh)
    p = np.full(kernels.n_snp, np.nan)
    valid = ok & np.isfinite(t)
    p[valid] = 2.0 * stats.t.sf(np.abs(t[valid]), kernels.df)
    return p, beta


def _null_one(mode: str, seed: int, kernels: Kernels, *,
              b0: np.ndarray, s2: float, chol: np.ndarray,
              ha0: float, hd0: float) -> float:
    rng = np.random.default_rng(seed)
    n = kernels.n
    if mode == "bootstrap":
        y = kernels.x @ b0 + np.sqrt(s2) * (chol @ rng.standard_normal(n))
    elif mode == "permutation":
        y = kernels.y[rng.permutation(n)]
    elif mode == "permutation_within":
        y = kernels.y.copy()
        for mother in np.unique(kernels.mothers):
            idx = np.where(kernels.mothers == mother)[0]
            y[idx] = kernels.y[rng.permutation(idx)]
    else:
        raise ValueError(f"unknown null mode: {mode}")
    ha, hd = fit_vc(y, kernels, starts=((ha0, hd0), (0.05, 0.05)))
    p, _ = scan(y, ha, hd, kernels)
    return float(np.nanmin(p))


def run(
    panel,
    blup: pd.DataFrame,
    trait: str,
    *,
    nrep: int = 500,
    modes=MODES,
    jobs: int = 1,
    seed: int = BASE_SEED,
    out_tag: str | None = None,
    verbose: bool = True,
) -> dict:
    d = build_hybrids(blup, panel.samples, conf=CONF)
    d = d.dropna(subset=[trait]).reset_index(drop=True)
    if verbose:
        print(f"Panel {panel.n_markers:,} SNP; n={len(d)}; trait={trait}", flush=True)

    kernels = build_kernels(panel, d, trait)
    t0 = time.time()
    ha0, hd0 = fit_vc(kernels.y, kernels, starts=((0.5, 0.2), (0.2, 0.5), (0.05, 0.05)))
    observed, beta = scan(kernels.y, ha0, hd0, kernels)
    valid = np.isfinite(observed)
    lam = float(np.median(stats.chi2.isf(observed[valid], 1)) / stats.chi2.ppf(0.5, 1))
    best = int(np.nanargmin(np.where(valid, observed, np.inf)))
    if verbose:
        print(
            f"REML: Ka={ha0:.3f}, Kd={hd0:.3f}, residual={1 - ha0 - hd0:.3f}; "
            f"lambda={lam:.2f}; tested {int(valid.sum()):,}; {time.time() - t0:.1f}s",
            flush=True,
        )
        print(
            f"best chr{int(panel.chrom[best])}:{int(panel.pos[best])} "
            f"p_dom={observed[best]:.3e} beta={beta[best]:+.3f}",
            flush=True,
        )

    a0 = ha0 * kernels.ka + hd0 * kernels.kd + (1.0 - ha0 - hd0) * np.eye(kernels.n)
    chol = np.linalg.cholesky(a0 + 1e-10 * np.eye(kernels.n))
    ai_x = np.linalg.solve(a0, kernels.x)
    b0 = np.linalg.solve(kernels.x.T @ ai_x, ai_x.T @ kernels.y)
    resid0 = kernels.y - kernels.x @ b0
    s2 = float((resid0 @ np.linalg.solve(a0, resid0)) / (kernels.n - kernels.x.shape[1]))

    seeds = np.random.SeedSequence(seed).spawn(nrep)
    nulls = {}
    for mode in modes:
        t0 = time.time()
        with threadpool_limits(limits=1):
            if jobs in (0, 1):
                minima = np.array([
                    _null_one(mode, int(s.spawn(1)[0].generate_state(1)[0]),
                              kernels, b0=b0, s2=s2, chol=chol, ha0=ha0, hd0=hd0)
                    for s in seeds
                ])
            else:
                minima = np.array(Parallel(n_jobs=jobs, backend="threading")(
                    delayed(_null_one)(
                        mode, int(s.spawn(1)[0].generate_state(1)[0]), kernels,
                        b0=b0, s2=s2, chol=chol, ha0=ha0, hd0=hd0,
                    )
                    for s in seeds
                ))
        nulls[mode] = minima
        if verbose:
            print(
                f"[{mode}] {nrep} replicates in {time.time() - t0:.0f}s; "
                f"5% threshold {np.quantile(minima, 0.05):.2e}",
                flush=True,
            )

    tag = out_tag if out_tag is not None else (
        "" if trait == "seed_weight_1000" else f"_{trait}"
    )
    write_csv(
        pd.DataFrame({"chr": panel.chrom, "pos": panel.pos,
                      "p_dom": observed, "beta_dom": beta}),
        f"dominance_scan{tag}.csv",
    )
    for mode, values in nulls.items():
        write_csv(pd.DataFrame({"min_p": values}), f"dominance_null_{mode}{tag}.csv")
    return {
        "scan": pd.DataFrame({"chr": panel.chrom, "pos": panel.pos,
                              "p_dom": observed, "beta_dom": beta}),
        "nulls": nulls,
        "ha": ha0,
        "hd": hd0,
        "lambda": lam,
        "best": int(best),
        "n": len(d),
    }


def canonical_vcf_sample(sample: str) -> str:
    """Map the numeric sample names of the separate VCF to the deposit names."""
    try:
        number = int(float(sample))
    except (TypeError, ValueError):
        return sample
    if 1 <= number <= 32:
        return f"LI{number}"
    if number == 33:
        return "LI34"
    if number == 34:
        return "LI35"
    if 35 <= number <= 52:
        return f"LI{number + 2}"
    return {53: "LI33", 54: "LI36", 55: "VA761", 56: "VK101", 57: "VK934"}.get(number, sample)


def panel_from_vcf(path) -> tuple:
    data = read_vcf_dosage(path)
    mapped = GenotypeData(
        dosage=data.dosage,
        chrom=data.chrom,
        pos=data.pos,
        samples=tuple(canonical_vcf_sample(s) for s in data.samples),
    )
    return filter_panel(mapped, None, call_rate=0.0, maf=0.05), mapped


def run_vcf(path, blup: pd.DataFrame, trait: str = "seed_weight_1000", **kwargs) -> dict:
    panel, data = panel_from_vcf(path)
    print(f"VCF panel: {panel.n_markers:,} SNP x {len(data.samples)} samples", flush=True)
    return run(panel, blup, trait, **kwargs)
