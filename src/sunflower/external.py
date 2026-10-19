"""Boundary to the external, well-tested tools used by the pipeline.

The project deliberately does **not** implement EMMAX, BLINK, PLINK or REML
from scratch.  Additive association scans are delegated to GEMMA (``-lmm 4``,
the score test that EMMAX popularised), LD pruning to PLINK 1.9
(``--indep-pairwise``), and mixed-model variance components to R/sommer.

Binaries are resolved through these environment variables, then ``PATH``:

    SUNFLOWER_GEMMA   (default: gemma)
    SUNFLOWER_PLINK   (default: plink1.9, then plink)
    SUNFLOWER_RSCRIPT (default: Rscript)

See ``docs/EXTERNAL_TOOLS.md`` for native installation instructions.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .genotypes import Panel


class ExternalToolError(RuntimeError):
    """Raised when a required native tool is missing or fails."""


def _which(env_var: str, *candidates: str) -> str:
    explicit = os.environ.get(env_var, "").strip()
    if explicit:
        if Path(explicit).is_file():
            return explicit
        raise ExternalToolError(f"{env_var}={explicit!r} is not a file")
    for candidate in candidates:
        found = shutil.which(candidate)
        if found:
            return found
    raise ExternalToolError(
        f"{candidates[0]} not found. Install it natively (see docs/EXTERNAL_TOOLS.md) "
        f"or set {env_var}."
    )


def gemma() -> str:
    return _which("SUNFLOWER_GEMMA", "gemma")


def plink() -> str:
    return _which("SUNFLOWER_PLINK", "plink1.9", "plink")


def rscript() -> str:
    return _which("SUNFLOWER_RSCRIPT", "Rscript")


def run(cmd: list[str | Path], *, cwd: str | Path | None = None,
        env: dict[str, str] | None = None, timeout: int | None = None) -> subprocess.CompletedProcess:
    """Run a native tool; on failure include its stderr in the exception."""
    full_env = os.environ.copy()
    full_env.setdefault("OMP_NUM_THREADS", "1")
    full_env.setdefault("OPENBLAS_NUM_THREADS", "1")
    full_env.setdefault("MKL_NUM_THREADS", "1")
    if env:
        full_env.update(env)
    proc = subprocess.run(
        [str(part) for part in cmd],
        cwd=None if cwd is None else str(cwd),
        env=full_env,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout,
    )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()[-4000:]
        raise ExternalToolError(
            f"{cmd[0]} exited {proc.returncode}: {' '.join(map(str, cmd))}\n{tail}"
        )
    return proc


def write_vcf(panel: Panel, path: str | Path) -> list[str]:
    """Write a biallelic VCF from the panel's 0/1/2 ALT dosage.

    Marker IDs are ``chrom:pos:index`` so they survive the PLINK round-trip
    and can be mapped back to rows of the panel.
    """
    path = Path(path)
    ids = [f"{c}:{p}:{i}" for i, (c, p) in enumerate(zip(panel.chrom, panel.pos))]
    gts = {0: "0/0", 1: "0/1", 2: "1/1"}
    with path.open("w", encoding="ascii") as fh:
        fh.write("##fileformat=VCFv4.2\n")
        for c in sorted({int(v) for v in panel.chrom if v > 0}):
            fh.write(f"##contig=<ID={c}>\n")
        fh.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t")
        fh.write("\t".join(panel.samples) + "\n")
        for j in range(panel.n_markers):
            row = [str(panel.chrom[j]), str(panel.pos[j]), ids[j],
                   "A", "T", ".", "PASS", ".", "GT"]
            for value in panel.dosage[j]:
                row.append("./." if not np.isfinite(value) else gts[round(value)])
            fh.write("\t".join(row) + "\n")
    return ids


@dataclass(frozen=True)
class PlinkPanel:
    """The three PLINK files plus the relatedness matrix GEMMA needs."""
    prefix: Path
    kinship: Path
    workdir: Path
    samples: tuple[str, ...]
    ids: tuple[str, ...]
    n_markers: int


def prepare_panel(panel: Panel, workdir: str | Path, prefix: str = "panel",
                  kinship: np.ndarray | None = None) -> PlinkPanel:
    """VCF -> PLINK bed/bim/fam and a relatedness matrix for GEMMA.

    If ``kinship`` is given (the shared VanRaden matrix), it is written and
    passed to GEMMA with ``-k``; otherwise GEMMA computes a standardised
    kinship with ``-gk 2``.  GEMMA's ``-gk`` reads the FAM phenotype column even
    when asked only for a kinship matrix, so a dummy ``0`` is written there.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    vcf = workdir / f"{prefix}.vcf"
    ids = write_vcf(panel, vcf)
    bed = workdir / prefix
    run([plink(), "--vcf", vcf, "--make-bed", "--out", bed,
         "--allow-extra-chr", "--silent"], cwd=workdir)

    fam = workdir / f"{prefix}.fam"
    rows = []
    for line in fam.read_text().splitlines():
        fields = line.split()
        fields[5] = "0"
        rows.append(" ".join(fields))
    fam.write_text("\n".join(rows) + "\n")

    if kinship is not None:
        kin_path = workdir / f"{prefix}_kin.txt"
        np.savetxt(kin_path, np.asarray(kinship, dtype=float), fmt="%.12g")
    else:
        kin_prefix = f"{prefix}_kin"
        # -gk 2 is the standardized (VanRaden-like) kinship.  The extra filters
        # keep GEMMA from silently dropping markers with >5% missing or MAF < 0.01.
        run([gemma(), "-bfile", bed, "-gk", "2", "-miss", "1", "-maf", "0",
             "-o", kin_prefix, "-outdir", workdir], cwd=workdir)
        # -gk 1 writes .cXX.txt (centred); -gk 2 writes .sXX.txt (standardised).
        kin_files = sorted(
            list(workdir.glob(f"{kin_prefix}.c*.txt"))
            + list(workdir.glob(f"{kin_prefix}.s*.txt"))
        )
        if not kin_files:
            raise ExternalToolError(
                f"GEMMA did not write a kinship matrix in {workdir}"
            )
        kin_path = kin_files[0]
    return PlinkPanel(
        prefix=bed,
        kinship=kin_path,
        workdir=workdir,
        samples=panel.samples,
        ids=tuple(ids),
        n_markers=panel.n_markers,
    )


def _write_phenotype(y: np.ndarray, path: Path) -> None:
    values = np.asarray(y, dtype=float)
    # GEMMA's missing value convention.
    text = "\n".join("-9" if not np.isfinite(v) else f"{v:.12g}" for v in values)
    path.write_text(text + "\n")


def _write_covariates(covariates: np.ndarray, path: Path) -> None:
    # GEMMA's -c reader expects one row per individual and no header; it adds
    # the intercept itself.
    covariates = np.atleast_2d(np.asarray(covariates, dtype=float))
    lines = ["\t".join(f"{v:.12g}" for v in row) for row in covariates]
    path.write_text("\n".join(lines) + "\n")


def gemma_log_value(path: Path, pattern: str) -> float | None:
    if not path.exists():
        return None
    match = re.search(pattern, path.read_text(errors="replace"))
    return float(match.group(1)) if match else None


def run_gemma(
    plink_panel: PlinkPanel,
    y: np.ndarray,
    *,
    covariates: np.ndarray | None = None,
    label: str = "scan",
) -> dict:
    """Run GEMMA's LMM Wald scan (REML) for one phenotype.

    ``-lmm 1`` is the Wald test with REML variance components, the closest
    GEMMA analogue of the original EMMAX t-test.  ``-lmm 4`` reports a Wald
    statistic based on MLE and was deliberately not used.

    ``-miss 1 -maf 0`` disables GEMMA's own extra filters: the panel filter is
    applied once, in Python, and must be identical across scans.
    """
    workdir = plink_panel.workdir
    pheno = workdir / f"{label}.pheno"
    _write_phenotype(y, pheno)
    cmd = [
        gemma(), "-bfile", plink_panel.prefix, "-k", plink_panel.kinship,
        "-lmm", "1", "-miss", "1", "-maf", "0",
        "-p", pheno, "-o", label, "-outdir", workdir,
    ]
    if covariates is not None:
        cov = workdir / f"{label}.cov"
        _write_covariates(covariates, cov)
        cmd += ["-c", cov]
    run(cmd, cwd=workdir)
    assoc_path = workdir / f"{label}.assoc.txt"
    assoc = pd.read_csv(assoc_path, sep="\t")
    assoc = assoc.rename(columns={"ps": "pos"})
    # GEMMA's p_wald is a z-test.  EMMAX reported a t-test with
    # df = n - covariates - 1; recompute that from GEMMA's beta/se so the
    # genomic-control lambda is on the same scale as the deposited tables.
    n_effective = int(np.isfinite(np.asarray(y, dtype=float)).sum())
    n_covariates = 1 + (0 if covariates is None else np.asarray(covariates).shape[1])
    degrees = max(n_effective - n_covariates - 1, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        statistic = assoc["beta"].to_numpy(float) / assoc["se"].to_numpy(float)
        assoc["p_t"] = 2.0 * stats.t.sf(np.abs(statistic), degrees)
    assoc["p"] = assoc["p_t"]
    if "p_score" not in assoc:
        assoc["p_score"] = np.nan
    assoc["marker_index"] = assoc["rs"].astype(str).str.rsplit(":", n=1).str[-1].astype(int)
    assoc = assoc.sort_values("marker_index").reset_index(drop=True)
    log = workdir / f"{label}.log.txt"
    return {
        "assoc": assoc,
        "pve": gemma_log_value(log, r"pve estimate[^\n=]*=\s*([0-9.eE+-]+)"),
        "pve_se": gemma_log_value(log, r"se\(pve\)[^\n=]*=\s*([0-9.eE+-]+)"),
        "n_analyzed": gemma_log_value(
            log, r"number of analyzed SNPs(?:/var)?\s*=\s*(\d+)"
        ),
    }


def plink_prune(
    plink_panel: PlinkPanel,
    *,
    window: int,
    step: int,
    r2: float,
    label: str = "prune",
) -> np.ndarray:
    """PLINK 1.9 ``--indep-pairwise`` on the prepared panel.

    Returns a boolean mask in panel order (True = kept).
    """
    workdir = plink_panel.workdir
    run([
        plink(), "--bfile", plink_panel.prefix,
        "--indep-pairwise", str(window), str(step), str(r2),
        "--out", label, "--silent",
    ], cwd=workdir)
    prune_in = workdir / f"{label}.prune.in"
    kept_ids = set(prune_in.read_text().splitlines())
    return np.array([marker_id in kept_ids for marker_id in plink_panel.ids], dtype=bool)


def run_rscript(script: str | Path, args: list[str | Path], *, cwd: str | Path | None = None,
                timeout: int | None = None) -> subprocess.CompletedProcess:
    return run([rscript(), script, *args], cwd=cwd, timeout=timeout)
