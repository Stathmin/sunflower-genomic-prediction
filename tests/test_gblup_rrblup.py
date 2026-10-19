import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

from sunflower.prediction import gblup_predict

pytestmark = pytest.mark.external


def test_gblup_matches_rrblup(tmp_path):
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript is required")
    check = subprocess.run(
        ["Rscript", "-e", "quit(status=!requireNamespace('rrBLUP', quietly=TRUE))"],
        capture_output=True, check=False,
    )
    if check.returncode:
        pytest.skip("R package rrBLUP is not installed")

    rng = np.random.default_rng(8)
    n = 25
    genotypes = rng.normal(size=(n, 15))
    kinship = genotypes @ genotypes.T / 15 + np.eye(n) * 0.1
    y = 2 * genotypes[:, 0] + rng.normal(size=n)
    design = np.ones((n, 1))
    np.savetxt(tmp_path / "K.csv", kinship, delimiter=",")
    np.savetxt(tmp_path / "y.txt", y)
    code = """
    library(rrBLUP)
    K <- as.matrix(read.csv("K.csv", header=FALSE))
    y <- scan("y.txt", quiet=TRUE)
    X <- matrix(1, nrow=length(y), ncol=1)
    fit <- mixed.solve(y, Z=NULL, K=K, X=X)
    yhat <- as.numeric(X %*% fit$beta) + as.numeric(fit$u)
    write.csv(data.frame(yhat=yhat), "fit.csv", row.names=FALSE)
    """
    subprocess.run(["Rscript", "-e", code], cwd=tmp_path, check=True, capture_output=True)
    expected = pd.read_csv(tmp_path / "fit.csv")["yhat"].to_numpy()
    ours = gblup_predict(y, design, kinship, kinship, design)
    assert np.corrcoef(ours, expected)[0, 1] > 0.999
    assert np.allclose(ours, expected, atol=1e-2, rtol=1e-3)
