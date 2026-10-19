import numpy as np
import pandas as pd

from sunflower.variance import baker_stats, derived_components, shares_and_se


def _row() -> pd.Series:
    values = {
        "s2_line": 0.02, "s2_ml": 0.05, "s2_yl": 0.03, "s2_yml": 0.02,
        "s2_e": 0.10, "se_line": 0.01, "se_ml": 0.02, "cov_line_ml": 0.0,
    }
    for i in range(1, 6):
        for j in range(1, 6):
            values[f"cov{i}{j}"] = 0.0
    return pd.Series(values)


def test_baker_ratio_and_se():
    result = baker_stats(_row())
    assert np.isclose(result["baker"], 2 * 0.02 / (2 * 0.02 + 0.05))
    assert result["se_baker"] > 0


def test_shares_sum_to_one():
    shares, errors = shares_and_se(_row())
    assert np.isclose(shares.sum(), 1.0)
    assert np.all(errors >= 0)


def test_derived_components_are_fractions():
    result = derived_components(_row(), ny=2, r=3)
    assert 0 < result["H2"] < 1
    assert 0 < result["plot_repeatability"] < 1
