import numpy as np

from sunflower.genotypes import (
    GenotypeData,
    chrom_number,
    filter_panel,
    hybrid_dosage,
    standardized_pca,
    vanraden_kinship,
)


def test_chrom_number():
    assert chrom_number("CM007890.2") == 1
    assert chrom_number("CM007906.2") == 17
    assert chrom_number("1") == 1
    assert chrom_number("scaffold") == -1


def test_filter_panel_imputes_and_subsets():
    dosage = np.array([
        [0.0, 1.0, 2.0, np.nan],
        [2.0, 2.0, 2.0, 2.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 2.0, 2.0],
    ])
    data = GenotypeData(
        dosage=dosage,
        chrom=np.array([1, 1, 2, 3]),
        pos=np.array([10, 20, 30, 40]),
        samples=("a", "b", "c", "d"),
    )
    panel = filter_panel(data, call_rate=0.5, maf=0.0)
    assert panel.n_markers >= 1
    assert not np.isnan(panel.dosage).any()
    assert panel.samples == ("a", "b", "c", "d")


def test_kinship_and_pca():
    rng = np.random.default_rng(0)
    dosage = (rng.random((50, 10)) > 0.5).astype(float)
    kinship = vanraden_kinship(dosage)
    assert kinship.shape == (10, 10)
    assert np.allclose(kinship, kinship.T)
    scores, ratio = standardized_pca(dosage, 3)
    assert scores.shape == (10, 3)
    assert ratio.shape == (3,)
    assert 0 < ratio[0] <= 1


def test_hybrid_dosage_is_mid_parent():
    dosage = np.array([[0.0, 2.0], [2.0, 0.0]])
    result = hybrid_dosage(dosage, ("m", "f"), ["m"], ["f"])
    assert result.shape == (1, 2)
    assert np.allclose(result, [[1.0, 1.0]])
