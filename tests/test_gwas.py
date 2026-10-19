import numpy as np
import pandas as pd

from sunflower.gwas import genomic_inflation, loci_table, scan_summary, top_hits


def test_genomic_inflation_is_near_one_for_uniform_p():
    p = np.random.default_rng(0).uniform(size=2000)
    assert 0.8 < genomic_inflation(p) < 1.2


def test_summary_and_top_hits():
    assoc = pd.DataFrame({
        "chr": [1, 1, 2],
        "pos": [10, 20, 30],
        "p": [0.5, 1e-9, 0.01],
        "beta": [0.1, 0.2, -0.1],
        "marker_index": [0, 1, 2],
    })
    summary = scan_summary(assoc, 3)
    assert summary["top"] == "chr1:20"
    assert summary["n_p<5e-8"] == 1
    hits = top_hits(assoc, 2)
    assert list(hits["pos"]) == [20, 30]


def test_loci_table_is_driven_by_the_input_file():
    assoc = pd.DataFrame({
        "chr": [1, 1, 2],
        "pos": [10, 20, 30],
        "p": [0.5, 1e-7, 0.01],
        "beta": [0.1, 0.2, -0.1],
        "marker_index": [0, 1, 2],
    })
    loci = pd.DataFrame({"locus": ["a"], "chr": [1], "pos": [15], "window": [10]})
    table = loci_table(assoc, loci)
    assert table.loc[0, "markers_in_window"] == 2
    assert table.loc[0, "present"]
