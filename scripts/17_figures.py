from sunflower.genotypes import load_genotypes
from sunflower.io import read_csv, read_parquet
from sunflower.names import is_line_id
from sunflower.paths import FIGURES

"""Figures 1-4 for the genomic analysis.

No new statistics are computed here: each panel is built from already
deposited tables and previously saved derived outputs, and the script
asserts that the recreated figures reproduce the reported summary numbers.

  Figure 1 - population structure: PCA of the 54 paternal lines on the
             filtered SNP panel, alongside line-level oil content.
             Checks: PC1 7.2 %, PC2 5.6 %.
  Figure 2 - genomic prediction: five-fold cross-validated hybrid-level
             accuracy and the sqrt(H2) ceiling, paternal-GCA-level accuracy
             under a leave-one-line-out scheme, and the corresponding
             learning curve.
  Figure 3 - parent genetic distance vs. hybrid seed yield, overall and by
             tester. Checks: r = 0.262 overall; 0.21 / 0.18 / 0.45 by
             tester; p = 0.0007 for VK934A.
  Figure 4 - genome-wide scan of dominance deviations for 1000-seed weight.
             Checks: exactly four SNPs below the empirical threshold of
             5.5e-6, lead SNP at Chr14:169211003, p = 1.17e-6, beta = +3.38 g.

Inputs (repository tables/genotypes):
  without_confectionery_lines.csv, prediction_leave_one_line_out.csv,
  prediction_accuracy_5fold.csv, hybrid_plot_level.parquet,
  line_phenotypes.parquet, merged_genotypes.npz, dominance_scan.csv.
Outputs: Figure1.png/.tiff .. Figure4.png/.tiff, written under ``FIGURES``.
"""
import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

FIGDIR = FIGURES
FIGDIR.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "axes.titlelocation": "left", "savefig.dpi": 600,
    "pdf.fonttype": 42, "legend.frameon": False})
MM = 1 / 25.4
OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "sky": "#56B4E9",
      "verm": "#D55E00", "purple": "#CC79A7", "yellow": "#F0E442", "grey": "#999999"}
TESTER = {"VK101A": OI["blue"], "VA761A": OI["green"], "VK934A": OI["orange"]}
CONF = ("LI29", "LI30")
MINUS = lambda s: s.replace("-", "\u2212")


def letter(ax, s, x=-0.02, y=1.02):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=10, fontweight="bold",
            ha="right", va="bottom")


def save(fig, name):
    for ext in ("png", "tiff"):
        kw = {"pil_kwargs": {"compression": "tiff_lzw"}} if ext == "tiff" else {}
        fig.savefig(FIGDIR / f"{name}.{ext}", dpi=600, bbox_inches="tight", **kw)
    plt.close(fig)
    from PIL import (
        Image,  # figure width after trimming margins must not exceed the 170 mm text column
    )
    wmm = Image.open(FIGDIR / f"{name}.png").size[0] / 600 * 25.4
    assert wmm <= 170, (name, round(wmm, 1))


# ------------------------------------------------ genotypes (SNP filtering as used elsewhere in the pipeline)
data = load_genotypes()
G = data.dosage
chrom = data.chrom
samp = list(data.samples)
fathers = [s for s in samp if is_line_id(s)]
assert len(fathers) == 54
fi = [samp.index(f) for f in fathers]


def _flt(cols):
    """call rate >= 0.9 and MAF >= 0.05 within the given samples, chromosomes 1-17 (Section 2.2)."""
    X = G[:, cols]
    with np.errstate(all="ignore", divide="ignore"):
        counts = np.sum(~np.isnan(X), 1)
        cr = counts / X.shape[1]
        af = np.divide(
            np.nansum(X, 1), 2 * counts,
            out=np.full(X.shape[0], np.nan), where=counts > 0,
        )
    maf = np.minimum(af, 1 - af)
    return (chrom >= 1) & (chrom <= 17) & (cr >= 0.9) & (maf >= 0.05)


_imp = lambda X: np.where(np.isnan(X), np.nanmean(X, 1, keepdims=True), X)
Gl = _imp(G[_flt(fi)][:, fi])                      # 54 paternal lines (Figure 1)
assert Gl.shape[0] == 81903, Gl.shape
Gk = _imp(G[_flt(list(range(len(samp))))])          # 57 parents (Figure 3), as for GBLUP
assert Gk.shape[0] == 82889, Gk.shape

# ---------------------------------------------------------------- Figure 1
Xf = Gl.T
Xc = Xf - Xf.mean(0); Xc /= (Xc.std(0) + 1e-9)
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
sg = np.sign(Vt.sum(1)); sg[sg == 0] = 1.0; U = U * sg
pcs = U[:, :2] * S[:2]
pev = (S ** 2 / np.sum(S ** 2))[:2] * 100
isconf = np.array([f in CONF for f in fathers])
L = read_parquet("line_phenotypes.parquet")
oil = L[L["trait"] == "oil_content"].groupby("genotype")["value"].mean().reindex(fathers)
assert oil.notna().all()

fig, axs = plt.subplots(1, 2, figsize=(170 * MM, 70 * MM), gridspec_kw={"width_ratios": [1.15, 1]})
ax = axs[0]
ax.scatter(pcs[~isconf, 0], pcs[~isconf, 1], s=16, color=OI["blue"], alpha=0.8, linewidths=0,
           label="oilseed lines (52)")
ax.scatter(pcs[isconf, 0], pcs[isconf, 1], s=34, marker="D", color=OI["orange"],
           edgecolor="black", linewidths=0.5, zorder=3, label="confectionery lines (2)")
_ci = sorted(np.where(isconf)[0], key=lambda i: pcs[i, 0])
for i, (dx, ha) in zip(_ci, ((-4, "right"), (4, "left"))):
    ax.annotate(fathers[i], (pcs[i, 0], pcs[i, 1]), xytext=(dx, 7), ha=ha,
                textcoords="offset points", fontsize=7)
ax.axhline(0, color="#E0E0E0", lw=0.5, zorder=0); ax.axvline(0, color="#E0E0E0", lw=0.5, zorder=0)
ax.set_xlabel(f"PC1 ({pev[0]:.1f}% of variance)")
ax.set_ylabel(f"PC2 ({pev[1]:.1f}% of variance)")
ax.set_title("Principal components of the 54 paternal lines")
ax.legend(loc="best")
letter(ax, "A", x=-0.1)
ax = axs[1]
bins = np.arange(26, 55, 1.0)
ax.hist(oil[~isconf], bins=bins, color=OI["blue"], alpha=0.85, label="oilseed")
ax.hist(oil[isconf], bins=bins, color=OI["orange"], alpha=0.95, label="confectionery")
ax.set_xlabel("Oil content of the line (%)")
ax.set_ylabel("Number of lines")
ax.set_title("Line oil content, 2021\u20132024 mean")
ax.legend(loc="upper left")
letter(ax, "B", x=-0.1)
fig.tight_layout(w_pad=2.0)
save(fig, "Figure1")

# ---------------------------------------------------------------- Figure 2
order = [("seed_weight_1000", "1000-seed\nweight"), ("oil_content", "Oil\ncontent"),
         ("oil_yield", "Oil\nyield"), ("seed_yield", "Seed\nyield")]
# Single-hybrid predictive ability, leave-one-line-out scheme (same scheme as for paternal GCA)
wt = read_csv("prediction_leave_one_line_out.csv").set_index("trait")
hyb = [wt.loc[t, "r_all_lofo"] for t, _ in order]
# Paternal-GCA-level accuracy (leave-one-line-out, from script 06)
gca = read_csv("prediction_gca_leave_one_out.csv").set_index("trait")
line = {trait: float(gca.loc[trait, "r"]) for trait in gca.index}
# Learning curve, 40 random draws per training-set size
NTR = [10, 20, 30, 40, 45]
curve = read_csv("learning_curve.csv")
LC, SD = {}, {}
for trait in ["seed_yield", "oil_content", "oil_yield", "seed_weight_1000"]:
    values = curve[curve["trait"] == trait].set_index("n_train").reindex(NTR)
    LC[trait] = values["r_mean"].to_numpy()
    SD[trait] = values["r_sd"].to_numpy()
TCOL = {"seed_weight_1000": OI["blue"], "oil_content": OI["green"], "oil_yield": OI["purple"],
        "seed_yield": OI["orange"]}
fig, axs = plt.subplots(1, 2, figsize=(152 * MM, 72 * MM), gridspec_kw={"width_ratios": [1.55, 1]})
ax = axs[0]
# Within-tester accuracy and the ceiling for the same hybrid set
c1 = read_csv("without_confectionery_lines.csv")
h2 = lambda tr, s: c1[(c1["trait"] == tr) & (c1["set"] == s)]["H2"].iloc[0]
ceil = [np.sqrt(h2("seed_weight_1000", "without LI29 and LI30")), np.sqrt(h2("oil_content", "without LI29 and LI30")), np.nan,
        np.sqrt(h2("seed_yield", "without LI29 and LI30"))]
within = [wt.loc[t, "r_within_mean_lofo"] for t, _ in order]
TC = ["r_VK101", "r_VA761", "r_VK934"]
lo = [wt.loc[t, TC].astype(float).min() for t, _ in order]; hi = [wt.loc[t, TC].astype(float).max() for t, _ in order]
x = np.arange(len(order)); w = 0.26
ax.bar(x - w, hyb, w, color="#0072B2", label="single hybrid, all hybrids")
ax.bar(x, within, w, color="#56B4E9", label="single hybrid, within tester")
ax.errorbar(x, within, yerr=[np.subtract(within, lo), np.subtract(hi, within)], fmt="none", ecolor="#333333",
            elinewidth=0.7, capsize=1.8)
ax.bar(x + w, [line[t] for t, _ in order], w, color="#BFE0F5", label="paternal GCA")
for k, (t, _) in enumerate(order):
    vals = [hyb[k], within[k], line[t]]
    ys = [hyb[k] + 0.015, max(within[k], hi[k]) + 0.015, line[t] + 0.015]
    for i in (1, 2):                        # keep each label at least 0.05 above the neighboring bar's label
        if abs(ys[i] - ys[i - 1]) < 0.05:
            ys[i] = ys[i - 1] + 0.05
    for xi, v, y in zip((x[k] - w, x[k], x[k] + w), vals, ys):
        ax.text(xi, y, f"{v:.2f}", ha="center", va="bottom", fontsize=6)
for xi, c in zip(x, ceil):
    if pd.notna(c):
        ax.plot([xi - w / 2, xi + w / 2], [c, c], color=OI["verm"], lw=1.4)
ax.plot([], [], color=OI["verm"], lw=1.4, label="upper bound \u221aH\u00b2 (within tester)")
ax.set_xticks(x, [e for _, e in order])
ax.set_ylim(0, 1.18); ax.set_yticks(np.arange(0, 1.01, 0.2))
ax.set_ylabel("Predictive ability (r)")
ax.set_title("Prediction for a paternal line held out of training")
ax.legend(loc="upper right", fontsize=6)
letter(ax, "A", x=-0.1)
ax = axs[1]

_off = {trait: offset for (trait, _), offset in zip(order, (-0.9, -0.3, 0.3, 0.9))}
for t, e in order:
    ax.errorbar(np.array(NTR) + _off[t], LC[t], yerr=SD[t], marker="o", ms=3.2, lw=1.1, elinewidth=0.7,
                capsize=1.6, color=TCOL[t], label=e.replace("\n", " "))
ax.set_xticks(NTR)
ax.set_ylim(-0.05, 1.0)
ax.set_xlabel("Paternal lines in the training set")
ax.set_ylabel("Predictive ability of paternal GCA (r)")
ax.set_title("Learning curve (mean \u00b1 SD, 40 draws)")
ax.legend(loc="upper left", fontsize=6.5)
letter(ax, "B", x=-0.1)
fig.tight_layout(w_pad=2.0)
save(fig, "Figure2")

# ---------------------------------------------------------------- Figure 3
MO = {"VK101A": "VK101", "VA761A": "VA761", "VK934A": "VK934"}
P = read_parquet("hybrid_plot_level.parquet")
hm = (P[P["trait"] == "seed_yield"].groupby(["mother", "father"])["value"].mean()
      .rename("yield").reset_index())
assert len(hm) == 162
s2c = {s: i for i, s in enumerate(samp)}
hm["dist"] = [np.mean(np.abs(np.round(Gk[:, s2c[MO[m]]]) - np.round(Gk[:, s2c[f]])) / 2)
              for m, f in zip(hm["mother"], hm["father"])]
r_all, p_all = stats.pearsonr(hm["dist"], hm["yield"])
rs = {}
fig, axs = plt.subplots(1, 3, figsize=(170 * MM, 60 * MM), sharey=True)
for ax, m, L_ in zip(axs, ["VK101A", "VA761A", "VK934A"], "ABC"):
    sub = hm[hm["mother"] == m]
    r, p = stats.pearsonr(sub["dist"], sub["yield"])
    rs[m] = r
    col = TESTER[m]
    ax.scatter(sub["dist"], sub["yield"], s=12, color=col, alpha=0.85, linewidths=0)
    bb = np.polyfit(sub["dist"], sub["yield"], 1)
    xs = np.array([sub["dist"].min(), sub["dist"].max()])
    ax.plot(xs, bb[0] * xs + bb[1], color="black", lw=1.0, ls="-" if p < 0.05 / 3 else "--")
    ptxt = "p < 0.0001" if p < 1e-4 else (f"p = {p:.3f}" if p < 0.05 else f"p = {p:.2f}")
    ax.set_title(f"{m}:  r = {r:.2f}, {ptxt}")
    letter(ax, L_, x=-0.06)
    ax.set_xlabel("Mother\u2013father distance")
axs[0].set_ylabel("Hybrid seed yield (t/ha)")
fig.tight_layout(w_pad=1.2)
# Heterogeneity test across the three within-tester correlations (unrounded r values)
zz = np.arctanh(list(rs.values())); Qd = (54 - 3) * ((zz - zz.mean()) ** 2).sum()
print(f"heterogeneity Q = {Qd:.2f}, p = {stats.chi2.sf(Qd, 2):.2f}")
save(fig, "Figure3")

# ---------------------------------------------------------------- Figure 4
sc = read_csv("dominance_scan.csv").dropna(subset=["p_dom"])
null = read_csv("dominance_null_bootstrap.csv")["min_p"].to_numpy()
THR = float(np.quantile(null, 0.05))   # empirical 5% threshold
hits = sc[sc["p_dom"] < THR]
sc = sc.sort_values(["chr", "pos"]).reset_index(drop=True)
sc["mlp"] = -np.log10(sc["p_dom"])
off = {}; start = 0; ticks = []
for c in range(1, 18):
    sub = sc[sc["chr"] == c]
    off[c] = start
    ticks.append(start + sub["pos"].max() / 2)
    start += sub["pos"].max() + 8e6
sc["gx"] = sc["pos"] + sc["chr"].map(off)
fig, axs = plt.subplots(1, 2, figsize=(170 * MM, 62 * MM), gridspec_kw={"width_ratios": [2.3, 1]})
ax = axs[0]
for c in range(1, 18):
    sub = sc[sc["chr"] == c]
    ax.scatter(sub["gx"], sub["mlp"], s=2.2, linewidths=0,
               color=OI["blue"] if c % 2 else "#8FB8D8", rasterized=True)
h = sc[sc["p_dom"] < THR]
ax.scatter(h["gx"], h["mlp"], s=16, color=OI["verm"], zorder=3, linewidths=0)
ax.axhline(-np.log10(THR), color=OI["verm"], lw=0.8, ls="--")
ax.text(sc["gx"].max(), -np.log10(THR) + 0.12, "empirical genome-wide 5% threshold",
        ha="right", va="bottom", fontsize=6.5, color=OI["verm"])
ax.set_xticks(ticks, [str(c) for c in range(1, 18)])
ax.set_xlim(-5e6, sc["gx"].max() + 5e6)
ax.set_xlabel("Chromosome")
ax.set_ylabel("\u2212log\u2081\u2080 p (dominance deviation)")
ax.set_title(f"Dominance scan, 1000-seed weight ({len(sc):,} tested SNPs, 156 hybrids)")
letter(ax, "A", x=-0.06)
ax = axs[1]
reg = sc[(sc["chr"] == 14) & (sc["pos"].between(168.2e6, 170.2e6))]
ax.scatter(reg["pos"] / 1e6, reg["mlp"], s=7, color=OI["blue"], linewidths=0)
hr = reg[reg["p_dom"] < THR]
ax.scatter(hr["pos"] / 1e6, hr["mlp"], s=18, color=OI["verm"], zorder=3, linewidths=0)
ax.axhline(-np.log10(THR), color=OI["verm"], lw=0.8, ls="--")
ax.annotate(f"{len(hits)} SNPs, {int(hits['pos'].min()):,}-{int(hits['pos'].max()):,}",
            (hr["pos"].min() / 1e6, hr["mlp"].max()),
            xytext=(168.42, 3.9), textcoords="data", fontsize=6.5, ha="left", va="top",
            arrowprops={"arrowstyle": "-", "lw": 0.5, "color": "#555555"})
ax.set_xlabel("Chr14 position (Mb)")
ax.set_title("Chr14, 168.2\u2013170.2 Mb")
letter(ax, "B", x=-0.1)
fig.tight_layout(w_pad=1.5)
save(fig, "Figure4")
print("figures written:", sorted(p.name for p in FIGDIR.iterdir()))
print("SNPs in scan:", len(sc))