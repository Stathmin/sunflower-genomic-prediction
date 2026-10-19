# External tools

The pipeline deliberately does **not** implement EMMAX, BLINK, PLINK or REML.
Those are delegated to maintained tools:

| Task | Tool | Used by |
|---|---|---|
| Additive LMM association (Wald / REML) | GEMMA 0.98.5 | scripts 10–13, 16 |
| LD pruning (`--indep-pairwise`) | PLINK 1.9 | script 19 |
| Variance components / Baker's ratio | R `sommer` | scripts 01–05, 16 |
| Independent GBLUP cross-check | R `rrBLUP` | tests |
| VCF reading | `pysam` | script 15 |

## Python environment

```bash
uv sync --extra dev          # runtime + pytest/ruff
uv run python scripts/06_gca_prediction.py
```

`uv.lock` is committed. Use `uv run` so the package in `src/sunflower` is on
`sys.path`; there is no import-time directory walking.

## Native tools

### With pixi (recommended for a single reproducible environment)

```bash
pixi install                 # plink, gemma, r-base
pixi run install-r           # sommer + rrBLUP
```

### With apt (Debian/Ubuntu)

```bash
sudo apt-get install plink1.9 gemma
Rscript src/sunflower/r/install.R
```

### With conda / mamba

```bash
conda env create -f environment.yml
conda activate sunflower-external
Rscript src/sunflower/r/install.R
```

The binaries are resolved through `PATH`, or explicitly through:

| Variable | Default |
|---|---|
| `SUNFLOWER_GEMMA` | `gemma` |
| `SUNFLOWER_PLINK` | `plink1.9`, then `plink` |
| `SUNFLOWER_RSCRIPT` | `Rscript` |

Check the versions with:

```bash
gemma -h | head -1
plink1.9 --version
Rscript -e 'cat(as.character(packageVersion("sommer")), "\n")'
Rscript -e 'cat(as.character(packageVersion("rrBLUP")), "\n")'
```

## BLINK

BLINK is **not bundled**. The old `scripts/models/blink.py` was a hand-written
approximation and was removed; reimplementing BLINK is out of scope. If a BLINK
comparison is needed, run the official BLINK implementation (for example through
GAPIT3) and merge its association table. The additive scans in this deposit use
GEMMA.

## Data root

The scripts read `tables/` and `genotypes/` from the current working directory.
Run them from the repository root, or set:

```bash
export SUNFLOWER_ROOT=/path/to/sunflower-genomic-prediction
```

The undeposited provenance inputs still use `SUNFLOWER_WORK` and
`SUNFLOWER_CALLING` (see `.env.example`).
