# How Many Paternal Lines Are Enough? Genomic Prediction of Combining Ability in a Sunflower Breeding Panel

Data and code for the article of the same title submitted to *Genes (MDPI)* by Mikhail Divashuk, Oksana Borisenko et al.

DOI of this archive: @@TODO[DOI]. Article: @@TODO[ARTICLE DOI].

The 54 paternal lines and three testers of the companion field study were genotyped by sequencing. The scripts predict the GCA of a held-out paternal line by GBLUP, relate parental genetic distance to hybrid yield, and run additive and dominance-deviation association scans. Hybrid phenotypes come from the companion field study (preprint: @@TODO[PREPRINT]); the tables needed here are included.

Raw GBS reads are available from the corresponding author on reasonable request.

## Layout

```
src/sunflower/  installed package: paths, I/O, genotypes, prediction, variance, external tools
scripts/        numbered analysis scripts (thin entry points)
tests/          pytest suite, including GEMMA/PLINK and rrBLUP cross-checks
docs/           EXTERNAL_TOOLS.md and REWORK.md
paths.py        compatibility shim for sunflower.paths
names.py        compatibility shim for sunflower.names
tables/         input tables and the tables written by the scripts
genotypes/      genotype call sets
scripts/provenance/  how the input tables were built from undeposited primary files
figures/        written by the figure script
logs/           pre-rework verification run (see docs/REWORK.md)
pyproject.toml  Python dependencies (uv)
uv.lock         locked Python environment
pixi.toml       native GEMMA/PLINK/R environment
```

## How to run

Python 3.11 or later; GEMMA, PLINK 1.9 and R with `sommer` are required by the
association, pruning and variance-component scripts. The Python side is managed
with uv:

```bash
uv sync --extra dev
uv run python scripts/06_gca_prediction.py
...
```

The native tools are installed with `pixi install` (or apt/conda), and the R
packages with `Rscript src/sunflower/r/install.R`; see
`docs/EXTERNAL_TOOLS.md`. The old `requirements.txt` was removed; `pyproject.toml`
and `uv.lock` are authoritative.

Scripts read `tables/` and `genotypes/` from the current directory and write to
`tables/` and `figures/`. Run them from the repository root, or set
`SUNFLOWER_ROOT`. Scripts that scan many panels/traits accept `--jobs`; the run
times below are from the original single-threaded verification run and are
superseded by the parallel runs. Each script is independent except where noted.

## Scripts

| Script | Writes | Reproduces | Run time |
|---|---|---|---|
| `01_hybrid_blups_biometry.py` | hybrid_blup_phenotypes_with_biometry.csv, hybrid_blup_biometry.txt | hybrid BLUPs of plant height and head diameter (input of script 14) | 13 s |
| `02_line_heritability.py` | line_heritability.csv, .txt | Table 2 (heritability of the line means) | 86 s |
| `03_variance_components.py` | variance_components.csv, .txt | H² of the hybrid means used as the upper bound √H² (Figure 2) | 6 min |
| `04_baker_ratio_intervals.py` | baker_ratio_intervals.json | profile-likelihood intervals of Baker's ratio (sommer) | 22 min (old) |
| `05_without_confectionery_lines.py` | without_confectionery_lines.csv, .txt | H² of the 156 hybrids with oilseed fathers (upper bound in Figure 2) | 38 min |
| `06_gca_prediction.py` | learning_curve.csv, prediction_gca_repeated_5fold.csv, prediction_gca_leave_one_out.csv | Table 1 and Figure 2B (all traits on the 52 oilseed lines): prediction of a held-out paternal line, the learning curve (mean and SD over draws) and repeated five-fold cross-validation (Section 3.2) | 4 s |
| `07_prediction_within_tester.py` | prediction_within_tester_5fold.csv, .txt | single-hybrid prediction within tester, 5-fold (Section 3.2) | 14 s |
| `08_prediction_leave_one_line_out.py` | prediction_leave_one_line_out.csv, .txt | single-hybrid prediction, leave-one-line-out (Section 3.2, Figure 2A) | 24 s |
| `09_parental_distance.py` | printed to the log | parental genetic distance and hybrid yield on the 82,889-SNP set, with a mixed model and paternal-line means allowing for the dependence among hybrids (Section 3.3, Figure 3) | 13 s |
| `10_lines_gwas_panels.py` | lines_gwas_panels.csv, lines_gwas_panels_scans.csv, lines_gwas_panels_hits.csv | GEMMA association scans on alternative marker panels; top hits; optional `--loci` annotation | 3 min (old) |
| `11_lines_gwas.py` | lines_gwas.csv | GEMMA LMM scans of the 52 oilseed lines (Section 3.4) | 68 s (old) |
| `12_use_type_artefact.py` | use_type_artefact.csv | GEMMA scans of the 54 lines with and without the confectionery lines, with a bootstrap threshold (Section 3.4) | 13 min (old) |
| `13_use_type_artefact_pc.py` | use_type_artefact_pc.csv | the same with principal-component covariates | 13 min (old) |
| `14_dominance_scan.py` | dominance_scan.csv, dominance_null_bootstrap.csv, dominance_null_permutation.csv, dominance_null_permutation_within.csv | two-kernel dominance-deviation scan for 1000-seed weight with three null distributions: parametric bootstrap, free permutation and permutation within tester (Section 3.5, Figure 4); `--jobs` parallelises the replicates | 25 min (500 replicates, old single CPU); `DOM_NREP=5000` reproduces the published nulls |
| `14_dominance_scan.py --trait oil_content` | dominance_*_oil_content.csv | the same scan for oil content (not significant) | 25 min |
| `14_dominance_scan.py --trait seed_yield` | dominance_*_seed_yield.csv | the same scan for seed yield (not significant) | 25 min |
| `14_dominance_scan.py --trait plant_height --phenotypes hybrid_blup_phenotypes_with_biometry.csv` | dominance_*_plant_height.csv | the same scan for plant height (not significant) | 21 min |
| `14_dominance_scan.py --trait head_diameter --phenotypes hybrid_blup_phenotypes_with_biometry.csv` | dominance_*_head_diameter.csv | the same scan for head diameter (not significant) | 21 min |
| `15_dominance_scan_no_imputation.py` | dominance_scan_no_imputation.csv, dominance_null_no_imputation.csv | Table S2: the Chr14 effect on the call set without missing genotypes | 5 min |
| `16_power_and_hybrid_lambda.py` | reported_numbers_check.csv | power, λ of the hybrid scans (GEMMA) and the reported numbers (sommer); recomputes the checks | 39 s |
| `17_figures.py` | figures/Figure1–4 (.png, .tiff) | Figures 1–4 | 15 s |
| `18_chr14_ld.py` | chr14_locus_ld.csv | linkage disequilibrium among the four SNPs of the Chr14 locus (Section 3.5) | 2 s |
| `19_ld_pruning.py` | ld_pruning_sensitivity.csv | GBLUP of paternal GCA after PLINK 1.9 `--indep-pairwise` pruning (Sections 2.5 and 3.2) | 9 s |
| `20_blup_within_fold.py` | prediction_gca_sensitivity.csv | leave-one-line-out prediction of paternal GCA with hybrid BLUPs refitted inside each fold, and with the nine removed plot values put back (Sections 3.2 and 4.5) | 2 min |

Run times are from the pre-rework verification run; scripts that scan many panels/traits now accept `--jobs`. Scripts that run a figure or reuse another script's output are run last: `05` needs `04`; `17` needs `05`, `06`, `08` and `14`; `18` needs `14`.

`scripts/provenance/` documents how the input tables were built from the breeders' field workbooks and from the two call sets before merging. These files are not deposited; the scripts are included so that each step can be read. To run them, set `SUNFLOWER_WORK` in a local `.env` file (see `.env.example`). These scripts keep the original Russian names of the undeposited files, folders and worksheets, and the Russian labels they match inside those files (for example `Фенотипы.xlsx`, sheet `Гибриды_делянки`), because the scripts must find them verbatim. All other scripts and tables are in English.

## Tables

| File | Rows × columns | Content |
|---|---|---|
| `baker_ratio_intervals.json` |  | profile-likelihood intervals of Baker's ratio |
| `chr14_locus_ld.csv` | 4 × 5 | pairwise r² among the four SNPs of the Chr14 locus |
| `dominance_null_bootstrap.csv` | 5000 × 1 | smallest p under the parametric-bootstrap null, 5000 replicates (1000-seed weight) |
| `dominance_null_bootstrap_head_diameter.csv` | 500 × 1 | its bootstrap null |
| `dominance_null_bootstrap_plant_height.csv` | 500 × 1 | its bootstrap null |
| `dominance_null_no_imputation.csv` | 400 × 1 | its null distribution |
| `dominance_null_permutation.csv` | 5000 × 1 | smallest p under free phenotype permutation, 5000 replicates (1000-seed weight) |
| `dominance_null_permutation_within.csv` | 5000 × 1 | smallest p under permutation within tester, 5000 replicates (1000-seed weight) |
| `dominance_null_permutation_head_diameter.csv` | 500 × 1 | its permutation null |
| `dominance_null_permutation_plant_height.csv` | 500 × 1 | its permutation null |
| `dominance_scan.csv` | 82889 × 4 | dominance-deviation scan, 1000-seed weight: chr, pos, p_dom, beta_dom |
| `dominance_scan_head_diameter.csv` | 82889 × 4 | dominance-deviation scan, head diameter |
| `dominance_scan_oil_content.csv`, `dominance_scan_seed_yield.csv` | 82889 × 4 | dominance-deviation scans, oil content and seed yield; their nulls are `dominance_null_{bootstrap,permutation,permutation_within}_<trait>.csv` (500 × 1 each) |
| `dominance_scan_no_imputation.csv` | 42289 × 4 | the scan on the call set without missing genotypes |
| `dominance_scan_plant_height.csv` | 82889 × 4 | dominance-deviation scan, plant height |
| `gca_fathers_as_delivered.csv` | 54 × 8 | paternal GCA from the delivered means |
| `gca_testers_as_delivered.csv` | 3 × 8 | tester GCA from the delivered means |
| `hybrid_blup_phenotypes.csv` | 162 × 5 | hybrid BLUPs of seed yield, oil content, 1000-seed weight and oil yield |
| `hybrid_blup_phenotypes_with_biometry.csv` | 162 × 7 | the same plus plant height and head diameter |
| `hybrid_plot_level.parquet` | 4492 × 6 | plot-level records of the 162 hybrids (2022–2023): year, mother (tester), father (line), replicate, trait, value |
| `hybrid_year_means_as_delivered.csv` | 159 × 11 | year means as delivered by the breeders (kept for comparison) |
| `hybrids_as_delivered.parquet` | 1908 × 9 | hybrid year means as delivered, long format |
| `line_heritability.csv` | 9 × 15 | heritability of the four-year line means |
| `line_phenotypes.parquet` | 7687 × 7 | line phenotypes 2021–2024, long format |
| `line_phenotypes_as_delivered.parquet` | 7688 × 7 | line phenotypes as delivered |
| `lines_gwas.csv` | 36 × 9 | GEMMA LMM scans of the lines |
| `lines_gwas_panels.csv` | 8 × 11 | marker panels compared in script 10 |
| `lines_gwas_panels_hits.csv` | 720 × 10 | top 10 markers per panel/trait (no hard-coded loci) |
| `lines_gwas_panels_scans.csv` | 72 × 9 | GEMMA scans on each panel |
| `prediction_accuracy_5fold.csv` | 4 × 6 | GBLUP predictive ability for single hybrids, 5-fold by paternal line |
| `prediction_gca_leave_one_out.csv` | 4 × 3 | leave-one-line-out paternal GCA accuracy used by Figure 2 (script 06) |
| `prediction_leave_one_line_out.csv` | 4 × 15 | single-hybrid predictive ability, leave-one-line-out |
| `prediction_within_tester_5fold.csv` | 4 × 14 | the same within each tester |
| `reported_numbers_check.csv` | 25 × 4 | numbers reported in the paper recomputed from the deposit |
| `use_type_artefact.csv` | 16 × 9 | scans with and without the confectionery lines |
| `use_type_artefact_pc.csv` | 16 × 9 | the same with principal components |
| `variance_components.csv` | 5 × 14 | variance components, Baker's ratio, H² and plot repeatability for five traits |
| `without_confectionery_lines.csv` | 10 × 19 | the same without the two confectionery lines, with the full set as control |
| `ld_pruning_sensitivity.csv` | 5 × 10 | GBLUP of paternal GCA after LD pruning (script 19) |
| `prediction_gca_sensitivity.csv` | 3 × 5 | predictive ability of paternal GCA under the main analysis, with BLUPs refitted inside each fold, and with the removed plot values put back (script 20) |
| `removed_plots.csv` | 9 × 6 | the nine plot values removed by the outlier screen (from the companion field-study repository) |

## Genotypes

| File | Content |
|---|---|
| `genotypes/union_call_set.vcf.gz` | union of the two variant-calling strategies before filtering, 57 samples |
| `genotypes/filt_geno_no_imputation.vcf` | separately assembled call set without missing genotypes (Table S2) |
| `genotypes/merged_genotypes.npz` | filtered genotype matrix used by the scripts: `genotypes` (0/1/2, −1 missing), `chrom`, `pos`, `samples` |

Sample names: LI1–LI54 are the paternal lines; VK101, VA761 and VK934 are the testers (their CMS A-forms).

## Names

Traits: `seed_yield` (t/ha), `oil_content` (%), `seed_weight_1000` (g), `oil_yield`, `plant_height` (cm), `head_diameter` (cm), `days_emergence_flowering`; line traits also `hull_content` (%), `autofertility_self`, `autofertility_open` and `days_central_to_side`. Testers: VK101A, VA761A, VK934A (in genotype files without the suffix A). Lines: LI1–LI54; LI29 and LI30 are the two confectionery lines.

## Verification

The logs in `logs/` are from the pre-rework verification run (26 September 2026).
The rework replaced the custom EMMAX/BLINK/PLINK/REML engines with GEMMA, PLINK
1.9 and R/sommer; the current verification results and the numerical differences
are in `docs/REWORK.md`. The variance-component, prediction and pruning outputs
still reproduce the deposit closely; the additive GWAS tables should be
regenerated and re-checked.

## License

Code: MIT License (`LICENSE`). Data (`tables/`, `genotypes/`): Creative Commons Attribution 4.0 International (CC BY 4.0).

## Contact

Mikhail Divashuk, All-Russia Research Institute of Agricultural Biotechnology, Moscow, divashuk@gmail.com
