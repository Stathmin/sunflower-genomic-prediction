import sys
from pathlib import Path as _PathBoot
_p = _PathBoot(__file__).resolve().parent
while _p != _p.parent and not (_p / "paths.py").exists():
    _p = _p.parent
if str(_p) not in sys.path:
    sys.path.insert(0, str(_p))
from paths import OUT, DEPOSIT, HERE as REPO_ROOT, use_models, en_table, en_ids, en_id
use_models()          # makes scripts/models importable: from baker_v4 import reml_fit, moments

"""Variance components, the Baker ratio, its profile-likelihood interval, plot
repeatability, and the heritability of the combination mean, recomputed
after excluding the two confectionery-type lines (LI29, LI30).

The REML fitting routine and the profile-likelihood interval routine are the
same ones used to produce the published five-trait variance-component table;
only the set of plot-level records changes (hybrids whose pollinator line is
LI29 or LI30 are excluded). As a control, the script first recomputes the
Baker ratio and H2 on the full set of combinations and checks that these
reproduce the published variance-component table and Baker-ratio intervals
before it recomputes both quantities on the reduced set.

A check that compared the input parquet file's checksum against a copy of it
kept in a separate history folder was dropped, because that history folder is
not part of this repository.

Input:
    tables/hybrid_plot_level.parquet   - plot-level phenotypes
    tables/variance_components.csv     - published variance components (control reference)
    tables/baker_ratio_intervals.json  - published Baker-ratio profile intervals (control reference)

Output:
    tables/without_confectionery_lines.csv
    tables/without_confectionery_lines.txt
"""
from pathlib import Path
import json
import importlib.util

import numpy as np
import pandas as pd

from baker_v4 import reml_fit, moments                            # noqa: E402

spec = importlib.util.spec_from_file_location(
    "baker_intervals", Path(__file__).resolve().parent / "04_baker_ratio_intervals.py"
)
BI = importlib.util.module_from_spec(spec)
spec.loader.exec_module(BI)

CONF = ("LI29", "LI30")
TRAITS = ["seed_yield", "oil_content", "seed_weight_1000", "plant_height", "head_diameter"]
WITH_CI = ("seed_yield", "oil_content", "seed_weight_1000")

H = (
    pd.read_parquet(OUT / "hybrid_plot_level.parquet")
    .rename(columns={"father": "line", "rep": "replicate"})
    .dropna(subset=["value"])
)


def one(d, with_ci):
    f = reml_fit(d, with_yl=True, hess=True)
    s = f["s2"]; e = f["s2e"]
    ny = float(d["year"].nunique())
    r = float(d.groupby(["year", "mother", "line"]).size().mean())
    G = s["line"] + s["tester_x_line"]
    GY = s["year_x_line"] + s["year_x_tester_x_line"]
    tot = G + GY + e
    out = dict(
        n=len(d),
        n_combinations=d.groupby(["mother", "line"]).ngroups,
        n_lines=d["line"].nunique(),
        plots_per_combination=r,
        s2_line=s["line"],
        s2_tester_x_line=s["tester_x_line"],
        s2_year_x_line=s["year_x_line"],
        s2_year_x_tester_x_line=s["year_x_tester_x_line"],
        s2_error=e,
        gca_share=s["line"] / tot,
        baker=f["baker"],
        se_baker=f.get("se_baker", np.nan),
        baker_moments=moments(d)["baker"],
        plot_repeatability=G / tot,
        H2=G / (G + GY / ny + e / (ny * r)),
        ci_low=np.nan,
        ci_high=np.nan,
    )
    if with_ci:
        m = BI.model(d)
        fit = m.fit(BI.STARTS)
        ci = BI.profile_interval(m, fit)
        out.update(ci_low=ci["low"], ci_high=ci["high"])
    return out


ref = pd.read_csv(OUT / "variance_components.csv").set_index("trait")
iv = json.loads((OUT / "baker_ratio_intervals.json").read_text(encoding="utf-8"))
rows, txt = [], []
for tr in TRAITS:
    full = H[H["trait"] == tr].reset_index(drop=True)
    a = one(full, tr in WITH_CI)
    # control: reproduce the published values
    assert abs(a["baker"] - ref.loc[tr, "baker_REML"]) < 1e-6, (tr, a["baker"], ref.loc[tr, "baker_REML"])
    assert abs(a["H2"] - ref.loc[tr, "H2_of_combination_mean"]) < 1e-6, tr
    if tr in WITH_CI:
        assert abs(a["ci_low"] - iv[tr]["ci_low"]) < 1e-6 and abs(a["ci_high"] - iv[tr]["ci_high"]) < 1e-6, tr
    sub = full[~full["line"].isin(CONF)].reset_index(drop=True)
    b = one(sub, tr in WITH_CI)
    for lab, x in (("all 162 combinations (control)", a), ("without LI29 and LI30", b)):
        rows.append(dict(trait=tr, set=lab, **x))
    txt += [
        f"### {tr}",
        f"  162 combinations (control, matches published values): baker {a['baker']:.4f} +/- {a['se_baker']:.4f}"
        f"  [{a['ci_low']:.3f}-{a['ci_high']:.3f}]  GCA share {a['gca_share']*100:.1f} %  H2 {a['H2']:.4f}"
        f"  plot repeatability {a['plot_repeatability']:.4f}",
        f"  without LI29/LI30 ({b['n_combinations']} combinations, {b['n_lines']} lines, {b['n']} plots): "
        f"baker {b['baker']:.4f} +/- {b['se_baker']:.4f}  [{b['ci_low']:.3f}-{b['ci_high']:.3f}]"
        f"  GCA share {b['gca_share']*100:.1f} %  H2 {b['H2']:.4f}  plot repeatability {b['plot_repeatability']:.4f}",
        f"  components without LI29/LI30: GCA {b['s2_line']:.4g}, SCA {b['s2_tester_x_line']:.4g}, "
        f"year_x_line {b['s2_year_x_line']:.4g}, year_x_tester_x_line {b['s2_year_x_tester_x_line']:.4g}, "
        f"error {b['s2_error']:.4g}",
        "",
    ]

pd.DataFrame(rows).to_csv(OUT / "without_confectionery_lines.csv", index=False, encoding="utf-8-sig")
(OUT / "without_confectionery_lines.txt").write_text("\n".join(txt), encoding="utf-8")
print("\n".join(txt))