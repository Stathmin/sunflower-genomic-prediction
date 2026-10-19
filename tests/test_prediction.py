import numpy as np
import pandas as pd

from sunflower.prediction import (
    cv_predictions,
    gblup_cv,
    hybrid_kinship,
    reml_variance,
)


def test_reml_and_leave_one_out():
    rng = np.random.default_rng(1)
    n = 30
    genotypes = rng.normal(size=(n, 20))
    kinship = genotypes @ genotypes.T / 20
    y = 2 * genotypes[:, 0] + rng.normal(size=n)
    s2g, s2e, _, _ = reml_variance(y, kinship, np.ones((n, 1)))
    assert s2g > 0 and s2e > 0
    predicted = np.array([
        gblup_cv(y, kinship, [j for j in range(n) if j != i], [i])[0]
        for i in range(n)
    ])
    assert np.corrcoef(predicted, y)[0, 1] > 0.3


def test_hybrid_kinship_shape():
    rng = np.random.default_rng(2)
    dosage = (rng.random((30, 8)) > 0.5).astype(float)
    samples = tuple(f"s{i}" for i in range(8))
    kinship = hybrid_kinship(dosage, samples, ["s0", "s1"], ["s2", "s3"])
    assert kinship.shape == (2, 2)
    assert np.allclose(kinship, kinship.T)


def test_cv_predictions_covers_every_hybrid():
    rng = np.random.default_rng(3)
    n_hybrids = 30
    mothers = [f"s{i % 3}" for i in range(n_hybrids)]
    fathers = [f"s{3 + i % 5}" for i in range(n_hybrids)]
    latent = rng.normal(size=(n_hybrids, 10))
    kinship = latent @ latent.T / 10 + np.eye(n_hybrids) * 0.1
    frame = pd.DataFrame({
        "mother": mothers, "father": fathers,
        "trait": 2 * latent[:, 0] + rng.normal(size=n_hybrids),
    })
    predicted = cv_predictions(frame, "trait", kinship, nfold=3)
    assert len(predicted) == n_hybrids
    assert predicted["yp"].notna().all()
