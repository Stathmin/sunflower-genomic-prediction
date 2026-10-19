# Rework notes

This branch replaces the hand-rolled infrastructure of the deposit with a small,
installed package and maintained external tools. The statistical models and the
reported pipeline order are unchanged; the engines behind the models are not.

## What was removed

| Before | After |
|---|---|
| 25 copies of a `while` loop that searched for `paths.py` and mutated `sys.path` | `src/sunflower/` package installed with `uv`; `SUNFLOWER_ROOT` is the only path setting |
| `scripts/models/emmax.py` plus five different inline EMMAX implementations | one GEMMA wrapper (`sunflower.external`) |
| `scripts/models/blink.py` (a hand-written approximation) | removed; BLINK is not reimplemented (see `docs/EXTERNAL_TOOLS.md`) |
| Hand-written PLINK `--indep-pairwise` in script 19 | PLINK 1.9 via `sunflower.external.plink_prune` |
| 361-line `baker_v4.REML` | R `sommer::mmer` via `sunflower.variance` |
| Hard-coded `KEY_LOCI` in script 10 | top-hit tables computed from each scan; optional `--loci` annotation file |
| Copy-pasted GBLUP / BLUP / panel-building code | `sunflower.prediction`, `sunflower.genotypes` |
| `allel` for the phase-6 VCF (not in the old requirements) | `pysam` |
| Single-CPU loops | `--jobs` with joblib/threads; GEMMA scans and bootstrap replicates run in parallel |

The obsolete modules `scripts/models/emmax.py`, `blink.py` and `baker_v4.py` were
deleted. `scripts/models/` no longer exists.

## Method changes and their numerical impact

### Additive association: GEMMA

- `scripts/10`, `11`, `12`, `13` and the hybrid check in `16` now call GEMMA
  `-lmm 1` (Wald test with REML variance components). This is the closest GEMMA
  analogue of EMMAX's t-test.
- The exact VanRaden kinship used by the old code is computed in
  `sunflower.genotypes.vanraden_kinship` and passed to GEMMA with `-k`.
- GEMMA's `-miss 1 -maf 0` are used so the marker filter is exactly the Python
  panel filter; otherwise GEMMA silently drops markers with >5% missing or
  MAF < 0.01.
- GEMMA's p-values are converted to a t-test with
  `df = n - covariates - 1`, matching the old EMMAX definition.
- **Impact:** lead SNPs and λ can differ because the variance-component
  estimator and test statistic are not identical. The panel sizes and the
  confectionery signal (27 hits with 54 lines vs 0 with 52 for oil content) are
  reproduced. The additive GWAS tables should be regenerated and the manuscript
  numbers re-checked.

### LD pruning: PLINK

`script 19` now writes the panel to PLINK bed/bim/fam and calls
`plink1.9 --indep-pairwise`. The no-pruning row reproduces the deposited
values exactly: 0.472 / 0.404 / 0.655 / 0.480.

### Variance components: R/sommer

`scripts/02–05` and `16` fit the same models with `sommer::mmer` (script 01
uses `statsmodels` MixedLM). The constrained profile in script 04 uses
`sommer`'s `overlay()` with
`r_tester_x_line = k * r_line`.

Reproduced on the deposit:

- Baker's ratio: 0.43957 vs 0.43951 (seed yield), 0.95772 vs 0.95772 (oil),
  0.95113 vs 0.95113 (1000-seed weight).
- Baker SE: 0.30569 vs 0.30571 (seed yield); 0.01645 vs 0.01639 (oil).
- H² of the combination mean: 0.42579 vs 0.42578 (seed yield), 0.89909 vs
  0.89909 (oil).
- Profile interval seed yield: 0–0.81504 vs 0–0.81506.
- Line heritability: plant height 0.948828 vs 0.948828; oil 0.86325 vs 0.86326.
- The method-of-moments `baker_moments` column is unchanged.

### Prediction

`sunflower.prediction` now owns the GBLUP and BLUP code used by scripts 06–08,
19 and 20. The leave-one-line-out values match the deposit exactly
(0.472 / 0.404 / 0.655 / 0.480), as do the LOFO single-hybrid values
(0.840 / 0.637 / 0.365 / 0.319) and the sensitivity checks in script 20.

One deliberate consistency change: hybrid allele frequencies for the hybrid
kinship are computed once on the full 156-hybrid set, not separately on each
trait-specific subset. This affects the within-tester correlations in script 07
slightly; the pooled five-fold controls still match the deposited table.

### Hard-coded loci

`KEY_LOCI` was removed from script 10. Each scan now writes
`tables/lines_gwas_panels_hits.csv` (top 10 markers per panel/trait). Known
loci can still be annotated explicitly:

```bash
uv run python scripts/10_lines_gwas_panels.py \
  --loci path/to/loci.csv      # columns: locus,chr,pos[,window]
```

`tables/lines_gwas_panels_loci.csv` was deleted because it was generated from
the hard-coded list.

## Verification performed

| Check | Result |
|---|---|
| `ruff check src scripts tests` | clean |
| `pytest -q` | 19 passed, 1 skipped (slow deposit check) |
| `uv lock` | lock file generated |
| Script 02 | H² and LR match the deposit |
| Script 03 | Baker, SE, H² and components match |
| Script 04 | profile intervals match to ~1e-4 |
| Script 06 | LOO and learning curve match exactly |
| Script 07/08 | pooled controls and LOFO values match |
| Script 10 | panel sizes (81,903/80,410, 100,827/99,201, 159,407/154,967, 30,988/30,079) match |
| Script 11 | runs; lead/λ differ from the custom EMMAX as documented above |
| Script 14 | observed lead chr14:169,211,003, p = 1.174e-6, beta = +3.377 (deposit: 1.17e-6, 3.38) |
| Script 15 | 42,289 SNPs after MAF; lead chr14:169,211,003 |
| Script 18 | four SNPs, r² = 1, 53 lines |
| Script 19 | no-prune values match exactly |
| Script 20 | sensitivity values match (0.469/0.404/0.655/0.478; 0.472/0.393/0.651/0.483) |

Full 200/500/5000-replicate null runs of scripts 12–14 were not repeated in the
review sandbox; the observed scans and the bootstrap mechanics were validated
with small `--nboot`/`--nrep` values. The production defaults are unchanged
(200, 500/5000, 400).

## Follow-ups

1. Regenerate the additive GWAS tables with the new GEMMA engine and re-check
   the manuscript's λ and lead-SNP statements.
2. Re-run the full bootstrap/null tables for scripts 12–14.
3. If a BLINK comparison is required, run the official implementation and add
   its table; do not resurrect the Python approximation.
