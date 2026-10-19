import os

import pytest

from sunflower.io import read_parquet
from sunflower.variance import baker_stats, hybrid_components

pytestmark = [pytest.mark.external, pytest.mark.slow]


def test_seed_yield_baker_ratio_matches_deposit():
    if os.environ.get("SUNFLOWER_RUN_SLOW") != "1":
        pytest.skip("set SUNFLOWER_RUN_SLOW=1 to run the deposit-level check")
    plot = read_parquet("hybrid_plot_level.parquet").rename(columns={"father": "line"})
    fit = hybrid_components(plot, ["seed_yield"], jobs=1)
    assert abs(baker_stats(fit.iloc[0])["baker"] - 0.4395) < 1e-3
