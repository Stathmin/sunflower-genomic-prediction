"""Genotype loading, filtering, kinship and PCA.

The deposit stores genotypes as a marker-by-sample ALT-dosage matrix
(0/1/2, -1 = missing) in ``genotypes/merged_genotypes.npz``.  Everything that
reads that matrix goes through this module so the chromosome mapping, missing
value convention and filters cannot drift between scripts.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .names import is_line_id
from .paths import GENOTYPES

_CHROM_RE = re.compile(r"CM00(\d+)\.2")


def chrom_number(name: str) -> int:
    """Map a deposit contig name to 1..17, or -1 if it is not a chromosome."""
    text = str(name)
    match = _CHROM_RE.search(text)
    if match:
        return int(match.group(1)) - 7889
    match = re.fullmatch(r"(?:chr)?(\d+)", text)
    if match:
        number = int(match.group(1))
        if 1 <= number <= 17:
            return number
    return -1


@dataclass(frozen=True)
class GenotypeData:
    dosage: np.ndarray          # markers x samples, NaN = missing
    chrom: np.ndarray           # marker chromosome number, -1 = unmapped
    pos: np.ndarray             # marker position, base pairs
    samples: tuple[str, ...]

    def column(self, sample: str) -> int:
        return self.samples.index(sample)

    @property
    def lines(self) -> list[str]:
        return [s for s in self.samples if is_line_id(s)]

    @property
    def testers(self) -> list[str]:
        return [s for s in self.samples if not is_line_id(s)]


@dataclass(frozen=True)
class Panel:
    dosage: np.ndarray          # markers x samples, mean-imputed
    raw: np.ndarray             # markers x samples before imputation
    chrom: np.ndarray
    pos: np.ndarray
    samples: tuple[str, ...]
    mask: np.ndarray            # boolean over the full marker set
    missing_fraction: float

    @property
    def n_markers(self) -> int:
        return self.dosage.shape[0]


def load_genotypes(path: str | Path | None = None) -> GenotypeData:
    path = Path(path) if path is not None else GENOTYPES / "merged_genotypes.npz"
    z = np.load(path, allow_pickle=True)
    dosage = z["genotypes"].astype(float)
    dosage[dosage < 0] = np.nan
    chrom = np.array([chrom_number(c) for c in z["chrom"]], dtype=int)
    pos = np.asarray(z["pos"], dtype=np.int64)
    samples = tuple(str(s) for s in z["samples"])
    return GenotypeData(dosage=dosage, chrom=chrom, pos=pos, samples=samples)


def impute_mean(dosage: np.ndarray) -> np.ndarray:
    """Replace missing calls by the marker mean; markers with no calls become 0."""
    mean = np.nanmean(dosage, axis=1, keepdims=True)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    return np.where(np.isnan(dosage), mean, dosage)


def filter_panel(
    data: GenotypeData,
    lines: list[str] | None = None,
    *,
    call_rate: float = 0.9,
    maf: float = 0.05,
    chromosomes: tuple[int, int] = (1, 17),
) -> Panel:
    """Apply the paper's marker filter to a set of samples.

    ``lines=None`` uses all 57 sequenced parents; otherwise only the requested
    samples are used for call rate, MAF and subsequent analysis.
    """
    samples = tuple(lines) if lines is not None else data.samples
    columns = [data.column(s) for s in samples]
    raw = data.dosage[:, columns]
    with np.errstate(invalid="ignore", divide="ignore"):
        counts = np.sum(~np.isnan(raw), axis=1)
        call = counts / raw.shape[1]
        total = np.nansum(raw, axis=1)
        af = np.divide(
            total, 2.0 * counts,
            out=np.full(raw.shape[0], np.nan), where=counts > 0,
        )
        minor = np.minimum(af, 1.0 - af)
    lo, hi = chromosomes
    mask = (
        (data.chrom >= lo)
        & (data.chrom <= hi)
        & (call >= call_rate)
        & (minor >= maf)
    )
    kept = raw[mask]
    return Panel(
        dosage=impute_mean(kept),
        raw=kept,
        chrom=data.chrom[mask],
        pos=data.pos[mask],
        samples=samples,
        mask=mask,
        missing_fraction=float(np.isnan(kept).mean()) if kept.size else 0.0,
    )


def subset_panel(panel: Panel, samples: list[str]) -> Panel:
    """Keep the panel's marker filter but restrict its columns to ``samples``."""
    columns = [panel.samples.index(sample) for sample in samples]
    raw = panel.raw[:, columns]
    return Panel(
        dosage=panel.dosage[:, columns],
        raw=raw,
        chrom=panel.chrom,
        pos=panel.pos,
        samples=tuple(samples),
        mask=panel.mask,
        missing_fraction=float(np.isnan(raw).mean()) if raw.size else 0.0,
    )


def vanraden_kinship(dosage: np.ndarray, jitter: float = 0.0) -> np.ndarray:
    """VanRaden genomic relationship matrix (markers x samples)."""
    p = dosage.mean(axis=1) / 2.0
    z = dosage - 2.0 * p[:, None]
    denom = 2.0 * float(np.sum(p * (1.0 - p)))
    if denom <= 0:
        raise ValueError("Cannot build a kinship matrix from monomorphic markers")
    kinship = (z.T @ z) / denom
    if jitter:
        kinship = kinship + np.eye(kinship.shape[0]) * jitter
    return kinship


def standardized_pca(dosage: np.ndarray, ncomp: int = 3):
    """PCA of samples after centring and standardising each marker.

    Returns ``(scores, variance_ratio)`` with scores ``n_samples x ncomp``;
    this is the convention used by the original EMMAX scripts and Figure 1.
    """
    x = dosage.T.astype(float)
    x = x - x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0)
    sd[sd < 1e-12] = 1.0
    x = x / sd
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    ratio = s**2 / np.sum(s**2)
    return u[:, :ncomp] * s[:ncomp], ratio[:ncomp]


def hybrid_dosage(dosage: np.ndarray, samples: tuple[str, ...],
                  mothers: list[str], fathers: list[str]) -> np.ndarray:
    """Mid-parent dosage for a list of hybrid combinations (hybrids x markers)."""
    index = {s: i for i, s in enumerate(samples)}
    mi = [index[m] for m in mothers]
    fi = [index[f] for f in fathers]
    return (dosage[:, mi] + dosage[:, fi]).T / 2.0


def read_vcf_dosage(path: str | Path) -> GenotypeData:
    """Read a biallelic VCF into the same dosage convention, using pysam."""
    import pysam

    with pysam.VariantFile(str(path)) as vcf:
        samples = tuple(vcf.header.samples)
        chrom, pos, rows = [], [], []
        for record in vcf:
            if len(record.alleles) < 2:
                continue
            row = []
            for sample in samples:
                call = record.samples[sample]
                indices = call.allele_indices
                if call.get("GT") is None or indices is None or None in indices:
                    row.append(np.nan)
                else:
                    # Number of non-reference alleles (0/1/2), the same
                    # convention as allel.GenotypeArray.to_n_alt().
                    row.append(float(sum(1 for allele in indices if allele > 0)))
            chrom.append(chrom_number(record.chrom))
            pos.append(record.pos)
            rows.append(row)
    return GenotypeData(
        dosage=np.asarray(rows, dtype=float),
        chrom=np.asarray(chrom, dtype=int),
        pos=np.asarray(pos, dtype=np.int64),
        samples=samples,
    )
