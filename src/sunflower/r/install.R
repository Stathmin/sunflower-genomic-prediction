#!/usr/bin/env Rscript
# Native R dependencies for the sunflower pipeline.
#
# sommer      variance components / REML (scripts 01-05, 16)
# rrBLUP      independent GBLUP cross-check (tests)
#
# BLINK is deliberately not installed here: the official implementation is a
# separate project, not a Python rewrite.  See docs/EXTERNAL_TOOLS.md.

packages <- c("sommer", "rrBLUP")
installed <- rownames(installed.packages())
missing <- setdiff(packages, installed)
if (length(missing)) {
  install.packages(missing, repos = "https://cloud.r-project.org")
}
for (pkg in packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop("R package not available: ", pkg)
  }
}
cat("R dependencies available:", paste(packages, collapse = ", "), "\n")
