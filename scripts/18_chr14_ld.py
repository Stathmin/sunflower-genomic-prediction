"""Linkage disequilibrium among the SNPs of the chromosome 14 dominance locus."""
from __future__ import annotations

import numpy as np
import pandas as pd

from sunflower.genotypes import load_genotypes
from sunflower.io import read_csv
from sunflower.names import is_line_id
from sunflower.paths import TABLES


def main() -> None:
    scan = read_csv("dominance_scan.csv").dropna(subset=["p_dom"])
    null = read_csv("dominance_null_bootstrap.csv")["min_p"].to_numpy()
    threshold = float(np.quantile(null, 0.05))
    hits = scan[scan["p_dom"] < threshold].sort_values("pos")
    assert set(hits["chr"]) == {14}, hits
    print(
        f"empirical 5% threshold {threshold:.2e}; SNPs below it: {len(hits)} "
        f"(chr14:{hits['pos'].min():,}-{hits['pos'].max():,})"
    )

    data = load_genotypes()
    indices = [
        int(np.where((data.chrom == 14) & (data.pos == int(position)))[0][0])
        for position in hits["pos"]
    ]
    genotype = data.dosage[indices]
    lines = [i for i, sample in enumerate(data.samples) if is_line_id(sample)]
    called = ~np.isnan(genotype[:, lines]).any(axis=0)
    r2 = np.corrcoef(genotype[:, lines][:, called]) ** 2
    labels = [f"chr14:{int(position)}" for position in hits["pos"]]
    table = pd.DataFrame(r2, index=labels, columns=labels)
    table.to_csv(TABLES / "chr14_locus_ld.csv", encoding="utf-8-sig")
    print(f"lines with all {len(indices)} SNPs called: {int(called.sum())} of {len(lines)}")
    print("pairwise r2:")
    print(table.round(4).to_string())
    print("genotype patterns over the lines (0/2 = homozygous, NaN = missing):")
    print(pd.DataFrame(genotype[:, lines].T, columns=labels).value_counts(dropna=False).to_string())
    print("Written: chr14_locus_ld.csv")


if __name__ == "__main__":
    main()
