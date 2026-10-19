# Verification run

> **Pre-rework log.** This file records the original custom-engine run. The
> rework (GEMMA / PLINK / R-sommer) and its verification are documented in
> `docs/REWORK.md`.

Date: 26 September 2026. Environment: Python 3.11.16; numpy 2.4.6, scipy 1.17.1, pandas 2.3.3, pyarrow 25.0.1, statsmodels 0.15.0, matplotlib 3.11.1, Pillow 12.3.0, openpyxl 3.1.5. Linux, 16 cores.

Every script in `scripts/` (not `scripts/provenance/`) was run on this repository; the logs are in this folder, and `run_summary.json` lists the exit status and run time of each. The tables the scripts wrote were compared with the tables from which the article was prepared.

| Output | Result |
|---|---|
| `variance_components.csv` | max relative difference 1.1e-06; column names differ (see below) |
| `without_confectionery_lines.csv` | max relative difference 2.0e-06 against the original calculation |
| `line_heritability.csv` | max relative difference 1.5e-07 |
| `hybrid_blup_phenotypes_with_biometry.csv` | max relative difference 3.1e-16 |
| `prediction_within_tester_5fold.csv` | max absolute difference 1.9e-07 against the original calculation |
| `prediction_leave_one_line_out.csv` | max absolute difference 5.4e-08 against the original calculation |
| `lines_gwas.csv` | max relative difference 1.4e-13; column names differ (see below) |
| `lines_gwas_panels.csv` | max relative difference 2.4e-14; column names differ (see below) |
| `use_type_artefact.csv`, `use_type_artefact_pc.csv` | λ and the number of SNPs at p < 5 × 10⁻⁸ identical in all 32 rows. In one row (54 lines with the confectionery lines, 1000-seed weight, no principal components), the permutation threshold is 2 × 10⁻⁶ instead of 3 × 10⁻⁶, and 37 SNPs pass it instead of 40. This count depends on the permutation draw and is not reported in the article. |
| `dominance_scan.csv` | max relative difference 1.1e-09; lead SNP Chr14:169,211,003, p = 1.17 × 10⁻⁶, bootstrap genome-wide p = 0.002 |
| `dominance_null_bootstrap.csv`, `dominance_null_permutation.csv` | max relative difference 9.5e-13 |
| `dominance_scan_plant_height.csv`, `…_head_diameter.csv` | best p = 6.52 × 10⁻⁶ and 1.18 × 10⁻⁴, genome-wide p = 0.058 and 0.689, as before |
| `reported_numbers_check.csv` | all 23 checked numbers agree with the article |
| `chr14_locus_ld.csv` | r² = 1 among the four SNPs in the 53 lines with all four called |
| `figures/Figure1–4.png` | pixel-identical to the figures of the submitted manuscript |

Column names changed from the earlier English tables: `se` (was `std_error`) in `variance_components.csv`; `h²_SNP` (was `h2_SNP`) in `lines_gwas.csv`; `dropped_cr`, `dropped_maf`, `bonferroni` (were `removed_cr`, `removed_maf`, `Bonferroni`) in `lines_gwas_panels.csv`. The scan tables also gained the columns `top` and `top_p` (the strongest SNP).

Changes of 3 October 2026 (manuscript edition 13). Figure 1 now uses the 81,903-SNP set of the 54 lines and the parental distance of Figure 3 and Section 3.3 the 82,889-SNP set of the 57 parents (previously a 161,791-SNP set with call rate ≥ 0.5). `06_gca_prediction.py` additionally writes `learning_curve.csv` (mean, SD and 2.5–97.5% range over the 40 draws; the means are unchanged) and `prediction_gca_repeated_5fold.csv`. Figures 1–4 regenerated from this repository are pixel-identical to those of the manuscript. The GBLUP of paternal GCA was cross-checked with rrBLUP 4.6.3 (kin.blup and mixed.solve, R 4.5.3): predictive abilities 0.426/0.421 (seed yield), 0.415/0.421 (oil content), 0.661/0.653 (1000-seed weight) and 0.599/0.595 (oil yield), against 0.426, 0.404, 0.655 and 0.596 here.

`19_ld_pruning.py` (added 3 October 2026): GBLUP of paternal GCA after LD pruning (windows of 50 and 1000 SNPs, r² > 0.5 and 0.2).


Changes of 8 October 2026 (manuscript edition 16). Scripts 06, 07, 08 and 19 analyse all four traits on the 156 hybrids of the 52 oilseed lines; previously seed yield and oil yield used all 162 hybrids. Paternal GCA, leave-one-line-out (seed yield, oil content, 1000-seed weight, oil yield): 0.472, 0.404, 0.655, 0.480; single hybrids over all hybrids 0.319, 0.637, 0.840, 0.365 and within tester 0.307, 0.339, 0.571, 0.333. `19_ld_pruning.py` reproduces the GCA values without pruning, and pruning gives 0.39–0.49, 0.37–0.42, 0.59–0.64 and 0.43–0.49. The stored control table `prediction_accuracy_5fold.csv` was updated to the 156 hybrids (seed yield 0.290, oil yield 0.350; H² and √H² of the oilseed hybrids from `without_confectionery_lines.csv`). `14_dominance_scan.py` gained a third null distribution, permutation within tester (environment variable `DOM_MODES` selects the nulls); for 1000-seed weight it gives the same genome-wide p = 0.014 as free permutation (`logs/14_dominance_scan__permutation_within.log`). The scan was run for oil content and seed yield (genome-wide p 0.48 and 0.29 by bootstrap, logs `14_dominance_scan__oil_content.log` and `__seed_yield.log`). The bootstrap and free-permutation tables of 1000-seed weight are those of the earlier run. GBLUP on the 52 lines was cross-checked with the R package rrBLUP 4.6.3 (`kin.blup`: 0.473, 0.415, 0.661, 0.481). `16_power_and_hybrid_lambda.py`: 25 checks, 0 mismatches. Figure 2 changed; Figures 1, 3 and 4 are unchanged, and all four are pixel-identical to edition 16.


Changes of 10 October 2026 (manuscript edition 18). The null distributions of the 1000-seed weight dominance scan were extended to 5000 replicates (`DOM_NREP=5000`): genome-wide p = 0.0096 under the parametric bootstrap (5% threshold 5.15 × 10⁻⁶) 0.0106 under permutation within tester and 0.0084 under free permutation; with 500 replicates these were 0.0020 (no replicate below the observed minimum), 0.0140 and 0.0140. The 500-replicate tables are kept in the manuscript folder `CLAUDE_FABLE_РЕЦЕНЗИЯ_DEEPSEEK_1Б_2026-10-10/депозит_до_ред18/`. The threshold line of Figure 4 moved from 5.5 × 10⁻⁶ to 5.2 × 10⁻⁶; the same four markers lie above it. `20_blup_within_fold.py` (new): refitting the hybrid BLUPs without the held-out line gives 0.469 / 0.404 / 0.655 / 0.478 (seed yield, oil content, 1000-seed weight, oil yield) against 0.472 / 0.404 / 0.655 / 0.480, and putting back the removed plot values gives 0.472 / 0.393 / 0.651 / 0.483. `09_parental_distance.py` now also fits a mixed model with paternal line as a random effect (p = 1.1 × 10⁻⁴) and the correlation over the 54 paternal-line means (r = 0.458, p = 0.0005). Of the 81,903 filtered markers 8,922 are short insertions or deletions; the lead marker of the chromosome 14 locus (169,211,003) is a single-base insertion. `16_power_and_hybrid_lambda.py`: 25 checks, 0 mismatches. Figures 1–3 unchanged; all four pixel-identical to edition 18.

Changes of 17 October 2026 (manuscript edition 20). Text-only edition: no table, figure or script output changed. `scripts/17_figures.py` now reads `tables/learning_curve.csv` through `paths.OUT` instead of an absolute path (figures re-generated, identical). Zenodo and citation metadata completed (affiliations, MIT licence).
