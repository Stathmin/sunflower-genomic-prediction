"""Linkage disequilibrium among the SNPs of the chromosome 14 dominance locus (Section 3.5).

The SNPs that pass the empirical 5% genome-wide threshold of the dominance-deviation scan for 1000-seed weight
(tables/dominance_scan.csv, threshold from tables/dominance_null_bootstrap.csv) are taken, and the squared
correlation r^2 between their genotype codes is computed over the paternal lines in which all of them are called.

Input:  tables/dominance_scan.csv, tables/dominance_null_bootstrap.csv, genotypes/merged_genotypes.npz
Output: tables/chr14_locus_ld.csv (pairwise r^2) and the printed genotype patterns
"""
import sys
from pathlib import Path as _PathBoot
_p = _PathBoot(__file__).resolve().parent
while _p != _p.parent and not (_p / "paths.py").exists():
    _p = _p.parent
if str(_p) not in sys.path:
    sys.path.insert(0, str(_p))
from paths import OUT, DEPOSIT

import numpy as np
import pandas as pd

scan = pd.read_csv(OUT / "dominance_scan.csv").dropna(subset=["p_dom"])
null = pd.read_csv(OUT / "dominance_null_bootstrap.csv")["min_p"].to_numpy()
thr = np.quantile(null, 0.05)
hits = scan[scan["p_dom"] < thr].sort_values("pos")
assert set(hits["chr"]) == {14}, hits
print(f"empirical 5% threshold {thr:.2e}; SNPs below it: {len(hits)} "
      f"(chr14:{hits['pos'].min():,}-{hits['pos'].max():,})")

z = np.load(DEPOSIT / "merged_genotypes.npz", allow_pickle=True)
chrom = np.array([str(c) for c in z["chrom"]])
order = list(dict.fromkeys(chrom))                      # contigs in chromosome order 1..17
c14 = order[13]
pos = z["pos"].astype(int)
samples = [str(s) for s in z["samples"]]
idx = [int(np.where((chrom == c14) & (pos == p))[0][0]) for p in hits["pos"]]
g = z["genotypes"][idx].astype(float)
g[g < 0] = np.nan
lines = [i for i, s in enumerate(samples) if s.startswith("LI")]
gl = g[:, lines]
ok = ~np.isnan(gl).any(axis=0)
r2 = np.corrcoef(gl[:, ok]) ** 2
labels = [f"chr14:{p}" for p in hits["pos"]]
pd.DataFrame(r2, index=labels, columns=labels).to_csv(OUT / "chr14_locus_ld.csv", encoding="utf-8-sig")
print(f"lines with all {len(idx)} SNPs called: {ok.sum()} of {len(lines)}")
print("pairwise r^2:")
print(pd.DataFrame(r2, index=labels, columns=labels).round(4).to_string())
print("genotype patterns over the lines (0/2 = homozygous, NaN = missing):")
print(pd.DataFrame(gl.T, columns=labels).value_counts(dropna=False).to_string())
print("Written: chr14_locus_ld.csv")
