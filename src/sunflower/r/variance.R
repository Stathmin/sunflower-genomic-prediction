#!/usr/bin/env Rscript
# Variance components via sommer::mmer (REML).
#
# Usage:
#   Rscript variance.R hybrid  input.csv output.csv
#   Rscript variance.R line    input.csv output.csv
#   Rscript variance.R profile input.csv output.csv "0,0.05,0.1,..."
#
# The Python wrapper (sunflower.variance) owns the model specification; this
# file only calls the well-tested sommer API.  No REML code is implemented here.

suppressPackageStartupMessages(library(sommer))
options(warn = -1)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) stop("usage: variance.R <hybrid|line|profile> input.csv output.csv [grid]")
mode <- args[[1]]
input <- args[[2]]
output <- args[[3]]

read_data <- function(path) {
  d <- read.csv(path, stringsAsFactors = FALSE)
  d$year <- factor(d$year)
  d$rep <- factor(d$rep)
  d$mother <- factor(d$mother)
  d$line <- factor(d$line)
  d$ml <- factor(paste(d$mother, d$line, sep = ":"))
  d
}

empty_row <- function(trait) {
  out <- data.frame(
    trait = trait, n = NA_integer_, converged = FALSE,
    s2_line = NA_real_, s2_ml = NA_real_, s2_yl = NA_real_,
    s2_yml = NA_real_, s2_e = NA_real_, aic = NA_real_,
    se_line = NA_real_, se_ml = NA_real_, se_yl = NA_real_,
    se_yml = NA_real_, se_e = NA_real_,
    cov_line_ml = NA_real_,
    stringsAsFactors = FALSE
  )
  for (i in 1:5) for (j in 1:5) out[[paste0("cov", i, j)]] <- NA_real_
  out
}

fit_hybrid <- function(d, trait) {
  out <- empty_row(trait)
  out$n <- nrow(d)
  fit <- tryCatch(
    mmer(
      value ~ year + year:rep + mother + year:mother,
      random = ~ vsr(line) + vsr(ml) + vsr(year:line) + vsr(year:mother:line),
      data = d, verbose = FALSE, getPEV = FALSE
    ),
    error = function(e) NULL
  )
  if (is.null(fit)) return(out)
  v <- unlist(fit$sigma)
  out$converged <- isTRUE(as.logical(fit$convergence)[1])
  out$s2_line <- v[[1]]
  out$s2_ml <- v[[2]]
  out$s2_yl <- v[[3]]
  out$s2_yml <- v[[4]]
  out$s2_e <- v[[5]]
  out$aic <- fit$AIC
  cov <- fit$sigmaSE
  if (!is.null(cov) && all(dim(cov) == c(5L, 5L))) {
    out$se_line <- sqrt(max(cov[1, 1], 0))
    out$se_ml <- sqrt(max(cov[2, 2], 0))
    out$se_yl <- sqrt(max(cov[3, 3], 0))
    out$se_yml <- sqrt(max(cov[4, 4], 0))
    out$se_e <- sqrt(max(cov[5, 5], 0))
    out$cov_line_ml <- cov[1, 2]
    for (i in 1:5) for (j in 1:5) out[[paste0("cov", i, j)]] <- cov[i, j]
  }
  out
}

fit_line <- function(d, trait) {
  d$genotype <- factor(d$genotype)
  d$year <- factor(d$year)
  out <- data.frame(
    trait = trait, n = nrow(d), n_lines = nlevels(droplevels(d$genotype)),
    n_years = nlevels(droplevels(d$year)), r_unit = NA_real_,
    s2g_pub = NA_real_, s2e_pub = NA_real_, aic_pub = NA_real_,
    s2g = NA_real_, s2gy = NA_real_, s2e = NA_real_, aic_full = NA_real_,
    converged = FALSE, stringsAsFactors = FALSE
  )
  out$r_unit <- nrow(d) / (out$n_lines * out$n_years)
  reduced <- tryCatch(
    mmer(value ~ year, random = ~ vsr(genotype), data = d,
         verbose = FALSE, getPEV = FALSE),
    error = function(e) NULL
  )
  full <- tryCatch(
    mmer(value ~ year, random = ~ vsr(genotype) + vsr(genotype:year), data = d,
         verbose = FALSE, getPEV = FALSE),
    error = function(e) NULL
  )
  if (!is.null(reduced)) {
    v <- unlist(reduced$sigma)
    out$s2g_pub <- v[[1]]
    out$s2e_pub <- v[[2]]
    out$aic_pub <- reduced$AIC
  }
  if (!is.null(full)) {
    v <- unlist(full$sigma)
    out$s2g <- v[[1]]
    out$s2gy <- v[[2]]
    out$s2e <- v[[3]]
    out$aic_full <- full$AIC
    out$converged <- isTRUE(as.logical(full$convergence)[1])
  }
  out
}

fit_profile <- function(d, trait, b) {
  out <- data.frame(trait = trait, b = b, aic = NA_real_, converged = FALSE)
  if (b <= 0) {
    random <- ~ vsr(ml) + vsr(year:line) + vsr(year:mother:line)
  } else if (b >= 1) {
    random <- ~ vsr(line) + vsr(year:line) + vsr(year:mother:line)
  } else {
    k <- 2 * (1 - b) / b
    random <- as.formula(sprintf(
      "~ vsr(overlay(line, ml, rlist = list(1, sqrt(%0.12g)))) + vsr(year:line) + vsr(year:mother:line)",
      k
    ))
  }
  fit <- tryCatch(
    mmer(value ~ year + year:rep + mother + year:mother, random = random,
         data = d, verbose = FALSE, getPEV = FALSE),
    error = function(e) NULL
  )
  if (!is.null(fit)) {
    out$aic <- fit$AIC
    out$converged <- isTRUE(as.logical(fit$convergence)[1])
  }
  out
}

if (mode == "hybrid") {
  d <- read_data(input)
  rows <- lapply(split(d, d$trait), function(part) fit_hybrid(part, part$trait[[1]]))
  write.csv(do.call(rbind, rows), output, row.names = FALSE)
} else if (mode == "line") {
  d <- read.csv(input, stringsAsFactors = FALSE)
  d$year <- factor(d$year)
  rows <- lapply(split(d, d$trait), function(part) fit_line(part, part$trait[[1]]))
  write.csv(do.call(rbind, rows), output, row.names = FALSE)
} else if (mode == "profile") {
  if (length(args) < 4) stop("profile mode needs a comma-separated grid")
  grid <- as.numeric(strsplit(args[[4]], ",", fixed = TRUE)[[1]])
  d <- read_data(input)
  trait <- d$trait[[1]]
  rows <- lapply(grid, function(b) fit_profile(d, trait, b))
  write.csv(do.call(rbind, rows), output, row.names = FALSE)
} else {
  stop("unknown mode: ", mode)
}
