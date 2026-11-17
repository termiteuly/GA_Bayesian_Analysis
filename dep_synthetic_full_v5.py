#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SIX-FREQUENCY DEP: reliability-aware GA/NUTS, interval scores and posterior recovery calibration.

Sampling modes
--------------
--sampling-mode stratified (legacy default): fixed truths per FIM stratum S1-S3.
--sampling-mode prior --prior-truths 60: sample independently from full broad
log-uniform prior with NO FIM selection. Include S4 in inference. FIM labels
are retained for POST-HOC description only. Truth-level discovery/validation
split is unconditional in prior mode.

Scientific question
-------------------
Across different LOCAL Fisher-information conditioning regimes, how do
posterior parameter recovery, precision, uncertainty coverage, and NUTS
computational diagnostics change under the SAME broad dielectric priors?

Method (single-shell dielectric theta, 4 dimensions; sigma_y inferred as fifth)
-----------------------------------------------------------------------------
1. Sample a FIXED pool of independent dielectric ground truths from the SAME
   independent log-uniform priors as the experimental DEP MCMC.
2. Compute the actual 4x4 Gaussian MEAN-model expected Fisher matrix in the
   coordinate phi = ln(theta), F = Jw.T @ Jw, using only the chosen frequencies.
   Evaluate kappa(F) via SVD(Jw): kappa(F) = [smax(Jw)/smin(Jw)]**2.
3. From a prespecified three-regime window (S1 <1e4, S2 1e4-1e8,
   S3 1e8-1e12), select N truths without replacement PER regime. S4 >=1e12
   stays in the prior screening report but NEVER enters GA/NUTS fitting.
   Stratification changes synthetic truth sampling but NOT inference priors.
4. Generate independently noisy individual DEP velocities per frequency with
   y[k,i] = v(f[k],theta_true) + sigma_true * N(0,1).
5. Fit exactly the SAME synthetic observations using (a) MULTISTART genetic
   algorithm for the four dielectric point estimates (Gaussian SSE/MLE) and
   (b) NumPyro NUTS for the joint posterior including unknown sigma_y.
6. Compare GA best-run point recovery with Bayesian posterior median recovery;
   retain all GA runs to inspect optimizer variability (NOT a confidence interval).
7. Save per-dataset and per-parameter metrics, GA trials, NUTS diagnostics and
   stratified summaries, allowing either method to resume independently.

IMPORTANT scientific limitations
--------------------------------
- The FIM is local, 4x4, and conditional on a Gaussian observation SD; sigma_y
  is INFERRED jointly during NUTS, not fixed there.
- A small FIM condition number alone does not guarantee large absolute
  information or accurate recovery; inspect lambda_min and posterior precision.
- The bins describe the selected synthetic population. Their unweighted total
  is NOT the broad-prior average. Nominal 95% posterior credible intervals are
  NOT guaranteed to have 95% coverage within FIM-selected strata.
- NUTS convergence DOES NOT guarantee identifiability or correct parameters.
- No model-misspecification or genuine biological variation is simulated.
- FIM intervals are LOCAL Gaussian approximations, NOT exact 95% coverage guarantees.
- CI/prior percentages are defined on log10 parameter scales; sigma_y has no finite prior range.
- ONLY the six actual experimental frequencies are used to compute FIM,
  simulate observations and fit GA/NUTS.
- A fixed common noise scale cancels in kappa(F), but DOES affect lambda_min,
  logdet(F), posterior precision, and recovery.

Updated 2026-10-31: full-vector joint posterior and dataset-level calibration
- Compute exact 4D simultaneous recovery from aligned NUTS draws.
- Preserve all original parameter-level analyses, GA/FIM/NUTS and reporting.
- Joint GA-centered and Bayesian-centered targets are validated separately.
- Joint report and discovery-selected reporting gate use DATASETS, not parameter counts.
- All old archives lacking full draws yield only partial Frechet bounds, NEVER fake joint mass.
- True physical/biological contrasts require specified target data, not automatic GA recovery.

Updated 2026-10-30: Posterior recovery benchmark (extension of v20261029)
- Retains the identical physical model, FIM strata and GA/NUTS procedures.
- New 95% normalized interval scores: Bayes, GA-local-FIM and prior reference.
- New truth-free posterior recovery probabilities centered on BOTH GA and Bayesian median.
- FIM-Gaussian recovery probabilities provide a fair same-target comparator for GA.
- Truth-level held-out calibration, Brier scores and fixed-threshold risk/reportability.
- Separate new outputs in output-dir/posterior_recovery/; original outputs retained.
- Re-run NUTS for GA-centered posterior probability unless original draws were saved.

Updated 2026-10-29:
- CLI default master --seed is 0 (explicit --seed recommended).
- CLI --dense-mass (default ON) and --no-dense-mass (diagonal baseline)
  change NumPyro NUTS kinetic-energy mass-matrix adaptation only.
- Both synthetic and experimental NUTS constructors use this setting.
- IMPORTANT: Old default scripts used seed=1204 and diagonal NUTS mass.
  To recreate old behavior run --seed 1204 --no-dense-mass (and preserve
  all other original options). Never --resume in an old output folder with
  a different mass setting; use a distinct output directory.

Requirements: numpy, pandas, matplotlib, scipy, jax, jaxlib, numpyro, arviz.
Without NumPyro, --dry-run and --ga-only perform screening / GA.

Default controlled manuscript study (60 independent truths, 540 paired datasets across 3 noise SD levels):
    python DEP_Reliability_Aware_GA_NUTS_PosteriorRecovery_v20261030.py --output-dir three_noise_6pt \
      --truths-per-stratum 20 --datasets-per-truth 3

Small pilot:
    python DEP_Reliability_Aware_GA_NUTS_PosteriorRecovery_v20261030.py --output-dir pilot_6pt \
      --truths-per-stratum 2 --datasets-per-truth 1 --candidate-draws 1000 \
      --ga-runs 3 --ga-population 40 --ga-generations 100 \
      --chains 2 --warmup 300 --samples 600

Screening-only:
    python DEP_Reliability_Aware_GA_NUTS_PosteriorRecovery_v20261030.py --output-dir dry_screen \
      --dry-run

Notes:
- --sigma-generation fixed (default) with --sigma-y 5.0 and
  --noise-multipliers 0.4142135623730951,1,2.414213562373095 gives
  sigma_true = HalfCauchy(5) Q25/Q50/Q75 = 2.071..., 5, 12.071... um/s.
- These are PRIOR QUANTILE scenarios, not experimentally validated noise levels.
- --spectra-plots representative makes paired DEP velocity / dielectric recovery plots.
- --truths-csv path/to/selected_truths.csv reuses prior truths, but re-evaluates FIM.
- --ga-only works without NumPyro, and --resume later adds NUTS fits.
- --sigma-generation prior is optional: each truth's noise SD is drawn
  from HalfCauchy(5) then multiplied by the chosen noise multiplier.
- --noise-multipliers 0.5,1.0,2.0 runs the SAME truths under three noise
  scales, sharing each truth/replicate's underlying standard Normal draws.
- To continue partial results, rerun with identical options plus --resume.
- --ga-only lets you finish GA before starting NumPyro NUTS.

"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

EPS0 = 8.854e-12
E_FIELD = 30000.0
VISCOSITY = 1e-3
RADIUS = 3e-6 + 55e-9
T_SHELL = 55e-9
EPS_MED = 80.0 * EPS0
SIG_MED = 0.1
DEP_CONST = ((RADIUS**2 * EPS_MED * (E_FIELD**2 / 9e-6)) / (3.0 * VISCOSITY)) * 1e6
PARAM_NAMES = ("eps_cyto", "sigma_cyto", "eps_shell", "sigma_shell")
Z_95 = 1.959963984540054
FACTOR2_LOG10 = math.log10(2)
PRIOR_SPAN_LOG10 = np.log10(np.array([20.,1e6,20.,1e6], dtype=np.float64))
LOG_NAMES = tuple("log_" + name for name in PARAM_NAMES)
LOW = np.array([10*EPS0, 1e-6, 10*EPS0, 1e-6], dtype=np.float64)
HIGH = np.array([200*EPS0, 1., 200*EPS0, 1.], dtype=np.float64)
LN_LOW = np.log(LOW)
LN_HIGH = np.log(HIGH)
FREQ_25 = np.array([
    5e4, 7.5e4, 1e5, 2e5, 3e5, 4e5, 5e5, 7.5e5,
    1e6, 2e6, 3e6, 4e6, 5e6, 7.5e6,
    1e7, 2e7, 3e7, 4e7, 5e7, 7.5e7,
    1e8, 2e8, 3e8, 4e8, 5e8
], dtype=np.float64)
FREQ_6 = np.array([5e5, 1e6, 2e6, 5e6, 1e7, 5e7], dtype=np.float64)
EXP_ANCHOR = np.array([9.3384e-10, 1.7548e-01, 2.7917e-10, 5.8656e-04])


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", type=Path, default=Path("fim_stratified_ga_nuts_results"))
    p.add_argument("--freq-grid", choices=["6"], default="6",
                   help="Experimental six-frequency design (only six frequencies allowed).")
    p.add_argument("--candidate-draws", type=int, default=10000,
                   help="Fixed unfiltered broad-prior screening sample size.")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--fim-bounds", default="1e4,1e8,1e12",
                   help="Three cutoffs: S1<first, S2 first-second, S3 second-third; S4>=third excluded.")
    p.add_argument("--truths-per-stratum", type=int, default=25)
    p.add_argument("--sampling-mode", choices=["stratified", "prior"], default="stratified",
                   help="stratified: equal S1/S2/S3 counts, exclude S4 (legacy); "
                        "prior: unconditioned log-uniform truth draws, retain S4 for analysis.")
    p.add_argument("--prior-truths", type=int, default=60,
                   help="Total independent log-uniform ground truths when --sampling-mode prior.")
    p.add_argument("--datasets-per-truth", type=int, default=3)
    p.add_argument("--n-per-frequency", type=int, default=50)
    p.add_argument("--sigma-generation", choices=["prior", "fixed"], default="fixed",
                   help="prior: true sigma from HalfCauchy(5), matching inference prior;"
                        " fixed: true sigma --sigma-y, inference still unknown.")
    p.add_argument("--sigma-y", type=float, default=5.0,
                   help="True sigma for fixed mode and common FIM screening reference.")
    p.add_argument("--noise-multipliers", default="0.4142135623730951,1,2.414213562373095",
                   help="Comma-separated positive multipliers for generating noise."
                        " Default levels are HalfCauchy(5) Q25,Q50,Q75 if sigma_y=5."
                        " Same standard-normal draws are paired across levels.")
    p.add_argument("--noise-multiplier", type=float, default=None,
                   help="Convenience alias for a single multiplier (default: 1.0).")
    p.add_argument("--truths-csv", type=Path, default=None,
                   help="Optional previous selected_truths.csv; reuse SAME truths and recompute FIM at current sigma reference.")
    p.add_argument("--spectra-plots", choices=["representative","all","none"], default="representative",
                   help="Representative: accurate and excellent-fit/poor-recovery cases for each stratum/noise.")
    p.add_argument("--ppc-draws", type=int, default=500,
                   help="NUTS posterior draws used for descriptive in-sample predictive checks.")
    p.add_argument("--posterior-curve-draws", type=int, default=200,
                   help="Posterior draws used to create dense-frequency latent predictive bands.")
    p.add_argument("--seed", type=int, default=0,
                   help="Master RNG seed for synthetic truths, noise, GA, and NUTS; explicitly set for reproducibility.")
    p.add_argument("--chains", type=int, default=3)
    p.add_argument("--chain-method", choices=["sequential", "parallel", "vectorized"], default="sequential")
    p.add_argument("--warmup", type=int, default=10000)
    p.add_argument("--samples", type=int, default=20000)
    p.add_argument("--target-accept", type=float, default=0.99)
    p.add_argument("--max-tree-depth", type=int, default=10)
    p.add_argument("--max-rhat", type=float, default=1.01)
    p.add_argument("--min-ess", type=float, default=1000.)
    p.add_argument("--dense-mass", action=argparse.BooleanOptionalAction, default=True,
                   help="Adapt a full HMC mass matrix (default: ON). Use --no-dense-mass for original diagonal baseline.")
    # Same GA settings as the original GA/Bayesian benchmark; multistart
    # minimizes the SAME individual-data Gaussian SSE used by NUTS likelihood.
    p.add_argument("--ga-runs", type=int, default=50)
    p.add_argument("--ga-population", type=int, default=100)
    p.add_argument("--ga-generations", type=int, default=500)
    p.add_argument("--ga-elite-fraction", type=float, default=0.10)
    p.add_argument("--ga-tournament-size", type=int, default=3)
    p.add_argument("--ga-crossover", type=float, default=0.80)
    p.add_argument("--ga-mutation", type=float, default=0.30)
    p.add_argument("--ga-log-mutation-sd", type=float, default=0.35)
    p.add_argument("--ga-early-stop", type=int, default=60)
    p.add_argument("--ga-local-polish", action=argparse.BooleanOptionalAction, default=True,
                   help="Use bounded L-BFGS-B after each GA run; raw GA estimate is also saved.")
    p.add_argument("--ga-only", action="store_true",
                   help="Fit GA only, no NumPyro needed. Can add NUTS later with --resume.")
    p.add_argument("--save-posteriors", action="store_true",
                   help="Also save 5D posterior arrays as compressed .npz files.")
    p.add_argument("--plot-representatives", action="store_true",
                   help="Plot posterior histograms and chain traces for the first dataset in each stratum.")
    p.add_argument("--backfill-joint", action="store_true",
                   help="With --resume, re-run NUTS only for records missing exact four-dimensional joint probability. Backs up old NUTS CSVs.")
    p.add_argument("--backfill-joint-limit", type=int, default=0,
                   help="Maximum NUTS backfills in this invocation (0=all). Try 2 first; continue with --resume.")
    p.add_argument("--joint-analysis-only", action="store_true",
                   help="Recalculate dataset-level JOINT posterior report from existing results; requires saved joint scalars or full posterior_samples NPZ.")
    p.add_argument("--joint-min-accepted-datasets", type=int, default=12,
                   help="Minimum accepted discovery datasets for selecting a joint gate.")
    p.add_argument("--joint-min-accepted-truths", type=int, default=5,
                   help="Minimum unique discovery truths contributing to selected joint gate.")
    p.add_argument("--audit-analysis-only", action="store_true",
                   help="Analyze existing evaluation_parameter_level.csv WITHOUT running GA/NUTS.")
    p.add_argument("--audit-target-factor", type=float, default=2.0,
                   help="Acceptable accuracy multiplier T>1; posterior P(T) requires saved NUTS draws if T!=2.")
    p.add_argument("--audit-target-risk", type=float, default=0.10,
                   help="Maximum empirical false-reassurance fraction on DISCOVERY data only.")
    p.add_argument("--audit-train-fraction", type=float, default=0.70,
                   help="Share of unique truths per FIM stratum allocated to discovery.")
    p.add_argument("--audit-min-accepted", type=int, default=24,
                   help="Minimum accepted parameter records on discovery for considering a rule.")
    p.add_argument("--audit-min-accepted-truths", type=int, default=8,
                   help="Minimum UNIQUE discovery truths contributing accepted parameter records.")
    p.add_argument("--audit-split-seed", type=int, default=24680,
                   help="Truth-level split seed, independent of GA/NUTS seed.")
    p.add_argument("--recovery-calibration-bins", type=int, default=10,
                   help="Fixed equally spaced probability bins for held-out calibration diagrams.")
    p.add_argument("--report-detail", choices=["core", "full"], default="core",
                   help="New reporting focus: core selected strategies or full legacy combinations. Does not change fitting.")
    p.add_argument("--report-quotas", default="0.50,0.60,0.75",
                   help="Predeclared accepted proportions for exploratory matched-reportability rankings.")
    p.add_argument("--audit-bootstrap", type=int, default=300,
                   help="Truth-cluster bootstrap repeats for validation uncertainty bands.")
    p.add_argument("--control-raw-csv", type=Path, default=None,
                   help="Existing long-format CSV input; XLSX input is also auto-detected by extension.")
    p.add_argument("--control-raw-xlsx", type=Path, default=None,
                   help="Use ORIGINAL six-column wide Excel sheet: row 1 = frequency headers (Hz); data below = velocities (um/s). Empty cells ignored.")
    p.add_argument("--control-sheet", type=str, default=None,
                   help="Optional XLSX worksheet name; default: first worksheet.")
    p.add_argument("--control-id", type=str, default=None,
                   help="Optional experiment identifier; default is the input CSV filename stem.")
    p.add_argument("--apply-control-csv", type=Path, default=None,
                   help="Apply previously DISCOVERY-calibrated rules to an unknown-truth audit CSV.")
    p.add_argument("--audit-rules-json", type=Path, default=None,
                   help="Rules JSON for --apply-control-csv; default output-dir/audit/selected_threshold_rules.json.")
    p.add_argument("--dry-run", action="store_true", help="Only FIM screen, select, and save; no NumPyro required.")
    p.add_argument("--no-auto-postprocess", action="store_true",
                   help="Skip automatic GA/Bayesian reporting and independent FIM diagnostics after synthetic run.")
    p.add_argument("--resume", action="store_true", help="Resume interrupted MCMC from stored CSVs.")
    a = p.parse_args()
    if a.backfill_joint_limit<0:
        p.error('--backfill-joint-limit must be >=0')
    if a.backfill_joint and (not a.resume or a.ga_only or a.joint_analysis_only or a.audit_analysis_only):
        p.error('--backfill-joint requires --resume and a NUTS-enabled synthetic fitting run')
    try:
        a.report_quotas=[float(z) for z in a.report_quotas.split(',')]
        if not a.report_quotas or any(not 0<z<=1 for z in a.report_quotas):
            p.error('--report-quotas must be comma-separated proportions in (0,1]')
    except ValueError:
        p.error('Invalid --report-quotas')
    try:
        a.fim_bounds = [float(v) for v in a.fim_bounds.split(",")]
        a.noise_multipliers = [float(v) for v in a.noise_multipliers.split(",")]
    except ValueError as exc:
        p.error(f"Bad comma-separated numerical argument: {exc}")
    if a.noise_multiplier is not None:
        if a.noise_multipliers != [0.4142135623730951,1.0,2.414213562373095]:
            p.error("Use either --noise-multiplier or --noise-multipliers, not both")
        a.noise_multipliers = [a.noise_multiplier]
    if len(a.fim_bounds) != 3:
        p.error("Exactly three --fim-bounds are required: S1/S2/S3 plus excluded S4")
    if not a.fim_bounds or any(not math.isfinite(x) or x <= 1 for x in a.fim_bounds):
        p.error("--fim-bounds must contain positive finite values > 1")
    if a.fim_bounds != sorted(set(a.fim_bounds)):
        p.error("--fim-bounds must be unique and strictly increasing")
    if not a.noise_multipliers or any(not math.isfinite(x) or x <= 0 for x in a.noise_multipliers):
        p.error("--noise-multipliers must contain finite positive numbers")
    if len(set(a.noise_multipliers)) != len(a.noise_multipliers):
        p.error("--noise-multipliers must not repeat values")
    for k in ["candidate_draws", "batch_size", "truths_per_stratum", "datasets_per_truth",
              "n_per_frequency", "chains", "warmup", "samples", "max_tree_depth", "posterior_curve_draws", "ppc_draws"]:
        if getattr(a,k) < 1:
            p.error(f"--{k.replace('_','-')} must be >=1")
    if not 0.5 < a.target_accept < 1:
        p.error("--target-accept must be between 0.5 and 1")
    if not (math.isfinite(a.sigma_y) and a.sigma_y > 0):
        p.error("--sigma-y must be finite and positive")
    if not 0 < a.max_rhat or a.min_ess < 0:
        p.error("Invalid diagnostic thresholds")
    for name in ("ga_runs", "ga_population", "ga_generations", "ga_tournament_size", "ga_early_stop"):
        if getattr(a, name)<1:
            p.error(f"--{name.replace('_','-')} must be >= 1")
    if a.ga_population < 4 or a.ga_tournament_size > a.ga_population:
        p.error("GA population must >= 4 and >= tournament size")
    for name in ("ga_crossover", "ga_mutation", "ga_elite_fraction"):
        if not 0 <= getattr(a, name) <= 1:
            p.error(f"--{name.replace('_','-')} must be between 0 and 1")
    if not 0 < a.ga_elite_fraction < 1 or a.ga_log_mutation_sd <= 0:
        p.error("GA elite fraction must (0,1), mutation log SD must >0")
    if a.prior_truths < 3:
        p.error("--prior-truths must be >=3 for information-tercile summaries")
    if a.sampling_mode == "prior" and a.candidate_draws < a.prior_truths and a.truths_csv is None:
        p.error("--candidate-draws must be >= --prior-truths in prior mode")
    if a.seed < 0:
        p.error("--seed must be nonnegative (required for reproducible SeedSequence streams)")
    if not (math.isfinite(a.audit_target_factor) and a.audit_target_factor>1):
        p.error("--audit-target-factor must be finite >1")
    if not (0<a.audit_target_risk<1 and 0<a.audit_train_fraction<1):
        p.error("--audit-target-risk and --audit-train-fraction must be in (0,1)")
    if a.recovery_calibration_bins<2 or a.recovery_calibration_bins>50:
        p.error("--recovery-calibration-bins must be 2..50")
    if a.audit_min_accepted<1 or a.audit_min_accepted_truths<1 or a.audit_bootstrap<0:
        p.error("--audit-min-accepted >=1, --audit-min-accepted-truths >=1, bootstrap >=0 required")
    if a.truths_csv is not None and not a.truths_csv.is_file():
        p.error("--truths-csv must point to an existing selected_truths.csv")
    return a


def complex_eps(f, ep, sigma):
    return ep - 1j*sigma/(2*jnp.pi*f + 1e-300)


def single_velocity(f, theta):
    """Verbatim single-shell physical model from the uploaded original scripts."""
    ec, sc, es, ss = theta[0], theta[1], theta[2], theta[3]
    ep_i = complex_eps(f, ec, sc)
    ep_o = complex_eps(f, es, ss)
    a = RADIUS/(RADIUS-T_SHELL)
    contrast = (ep_i-ep_o)/(ep_i+2*ep_o+1e-300)
    ep_p = ep_o*((a**3+2*contrast)/(a**3-contrast+1e-30j))
    ep_med = complex_eps(f, EPS_MED, SIG_MED)
    fcm = (ep_p-ep_med)/(ep_p+2*ep_med+1e-300)
    return DEP_CONST*jnp.real(fcm)


def make_forward_and_jac(freqs):
    f = jnp.asarray(freqs, dtype=jnp.float64)
    def log_velocity(logtheta):
        return single_velocity(f, jnp.exp(logtheta))
    pred = jax.jit(log_velocity)
    jac = jax.jit(jax.vmap(jax.jacfwd(log_velocity)))
    return pred, jac


def svd_information(jac, n_per_freq, sigma):
    """Actual 4x4 FIM, stable singular-value spectrum, and information measures."""
    jac = np.asarray(jac, dtype=np.float64)
    counts=np.asarray(n_per_freq,dtype=float)
    if counts.ndim==0:
        counts=np.full(jac.shape[0],float(counts))
    if counts.shape!=(jac.shape[0],) or np.any(~np.isfinite(counts)) or np.any(counts<=0):
        raise ValueError("n_per_freq must be a positive count per frequency (or one positive scalar).")
    jw = jac*(np.sqrt(counts)/float(sigma))[:,None]
    F = jw.T@jw
    F = 0.5*(F+F.T)
    sv = np.linalg.svd(jw, compute_uv=False)
    eigs = sv[::-1]**2  # F eigenvalues from singular values (avoid Gram matrix cancellation)
    if sv[-1] == 0:
        cond_j = float("inf")
        log10_cond_f = float("inf")
    else:
        cond_j = float(sv[0]/sv[-1])
        log10_cond_f = float(2.0*(np.log10(sv[0])-np.log10(sv[-1])))
    cond_f = float(10**log10_cond_f) if log10_cond_f < 308 else float("inf")
    logdet = float(2*np.sum(np.log(sv))) if np.all(sv>0) else float("-inf")
    # Four-parameter log-coordinate information diagnostics.  All depend on
    # sigma except condition number.  Geometric eigenvalue mean is the 4th
    # root of det(F); it is NOT a substitute for weakest-direction information.
    log10_min = 2*float(np.log10(sv[-1])) if sv[-1]>0 else float("-inf")
    log10_max = 2*float(np.log10(sv[0])) if sv[0]>0 else float("-inf")
    log10_geom = float(2*np.mean(np.log10(sv))) if np.all(sv>0) else float("-inf")
    log10_trace = float(np.logaddexp.reduce(2*np.log(sv))/np.log(10)) if np.all(sv>0) else float("-inf")
    return dict(F=F, Jw=jw, sv=sv, eigs=eigs, cond_j=cond_j,
                fim_cond=cond_f, log10_fim_cond=log10_cond_f,
                fim_lmin=float(eigs[0]), fim_lmax=float(eigs[-1]),
                log10_lambda_min=log10_min, log10_lambda_max=log10_max,
                log10_fim_info_geom=log10_geom, log10_fim_trace=log10_trace,
                fim_logdet=logdet, fim_log10det=log10_geom*4,
                fim_trace=float(np.sum(eigs)),
                rank_tol=float(np.finfo(float).eps*max(jw.shape)),
                rank_numeric=int(np.sum(sv>sv[0]*np.finfo(float).eps*max(jw.shape))))


def stratum_labels(bounds, include_s4=False):
    """Diagnostic FIM bins; S4 is INCLUDED only for unconditional prior mode."""
    assert len(bounds) == 3, "Exactly three regime cutoffs required"
    labels = ["S1_low", "S2_moderate", "S3_high"]
    if include_s4:
        labels.append("S4_extreme")
    return [1.0] + list(bounds), labels


def label_bin(cond, edges, labels):
    """Keep S4 visible in screening, but never select it for fitting."""
    if np.isnan(cond):
        return "invalid"
    if cond >= edges[-1]:
        return "S4_extreme" if "S4_extreme" in labels else "S4_extreme_excluded"
    if cond < edges[0]:
        return "invalid"
    for i,label in enumerate(labels):
        if edges[i] <= cond < edges[i+1]:
            return label
    return "invalid"


def label_information(log10_lambda_min, cuts):
    """Information classes using stored REFERENCE-SIGMA log10(lambda_min) terciles."""
    if not np.isfinite(log10_lambda_min):
        return "I_invalid"
    q1,q2=cuts
    return ("I1_low" if log10_lambda_min<q1 else
            "I2_medium" if log10_lambda_min<q2 else "I3_high")


def safe_csv_save(df, path):
    tmp = Path(str(path)+".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


# =============================================================================
# MULTISTART GA POINT ESTIMATOR (single-shell; exactly same observations as NUTS)
# =============================================================================

def ga_velocity_numpy(freqs, xlog):
    """Vectorized NumPy forward solver, same equations/constants as single_velocity.

    xlog can be (4,) or (population,4). No JAX compilation on each GA fitness.
    """
    x = np.asarray(xlog, dtype=float)
    one = (x.ndim == 1)
    if one:
        x = x[None, :]
    theta = np.exp(x)
    f = np.asarray(freqs, dtype=float)[None, :]
    ec, sc, es, ss = (theta[:, i, None] for i in range(4))
    ei = ec - 1j * sc / (2*np.pi*f + 1e-300)
    eo = es - 1j * ss / (2*np.pi*f + 1e-300)
    a = RADIUS / (RADIUS-T_SHELL)
    contrast = (ei-eo)/(ei+2*eo+1e-300)
    ep = eo * ((a**3+2*contrast)/(a**3-contrast+1e-30j))
    medium = EPS_MED - 1j*SIG_MED/(2*np.pi*f+1e-300)
    fcm = (ep-medium)/(ep+2*medium+1e-300)
    v = DEP_CONST*np.real(fcm)
    return v[0] if one else v


def ga_fitness(xlog, freqs, sample_means, n_per_freq):
    """SSE of individual observations up to theta-independent within-cell SSE.

    Sum_{k,i}(y_ik - v_k)^2 = sum_{k,i}(y_ik - ybar_k)^2
                                + n * sum_k (ybar_k-v_k)^2.
    """
    pred = ga_velocity_numpy(freqs, xlog)
    delta = pred - np.asarray(sample_means)
    # With unequal counts, each frequency mean receives the correct n_k weight.
    weights=np.asarray(n_per_freq,dtype=float)
    val = np.sum(weights * delta**2, axis=-1)
    return np.where(np.isfinite(val),val,np.inf)


def one_ga_start(a, freq, mean, seed, n_per_freq=None):
    from scipy.optimize import minimize
    if n_per_freq is None:
        n_per_freq=a.n_per_frequency
    rng = np.random.default_rng(seed)
    pop = rng.uniform(LN_LOW,LN_HIGH,size=(a.ga_population,4))
    scores = ga_fitness(pop,freq,mean,n_per_freq)
    bi = int(np.argmin(scores))
    best_x=pop[bi].copy()
    best_score=float(scores[bi])
    stall=0
    generations=0
    for generation in range(1,a.ga_generations+1):
        generations=generation
        elite_count=max(1,int(a.ga_elite_fraction*a.ga_population))
        elite=pop[np.argsort(scores)[:elite_count]].copy()
        children=[]
        while len(children)<a.ga_population-elite_count:
            ix1=rng.integers(0,a.ga_population,size=a.ga_tournament_size)
            ix2=rng.integers(0,a.ga_population,size=a.ga_tournament_size)
            p1=pop[ix1[np.argmin(scores[ix1])]]
            p2=pop[ix2[np.argmin(scores[ix2])]]
            if rng.random()<a.ga_crossover:
                mask=rng.random(4)<0.5
                midpoint=0.5*(p1+p2)
                c1=np.where(mask,p1,midpoint)
                c2=np.where(~mask,p2,midpoint)
            else:
                c1,c2=p1.copy(),p2.copy()
            for child in (c1,c2):
                if rng.random()<a.ga_mutation:
                    child=child+rng.normal(0,a.ga_log_mutation_sd,size=4)
                children.append(np.clip(child,LN_LOW,LN_HIGH))
                if len(children)>=a.ga_population-elite_count:
                    break
        pop=np.vstack((elite,np.array(children)))
        scores=ga_fitness(pop,freq,mean,n_per_freq)
        bi=int(np.argmin(scores))
        score=float(scores[bi])
        if score+1e-12<best_score:
            best_x=pop[bi].copy()
            best_score=score
            stall=0
        else:
            stall+=1
        if stall>=a.ga_early_stop:
            break
    final_x=best_x.copy()
    final_score=best_score
    polished=False
    if a.ga_local_polish:
        res=minimize(lambda xx: float(ga_fitness(xx,freq,mean,n_per_freq)),
                     final_x,method="L-BFGS-B",bounds=list(zip(LN_LOW,LN_HIGH)),
                     options={"maxiter":350,"ftol":1e-13})
        if np.isfinite(res.fun) and float(res.fun)<final_score:
            final_x=np.asarray(res.x,dtype=float)
            final_score=float(res.fun)
            polished=True
    return dict(ga_seed=int(seed),ga_generations=int(generations),
                ga_polish_improved=bool(polished),
                ga_sse_mean_raw=best_score,ga_sse_mean_final=final_score,
                **{f"ga_raw_{name}":float(np.exp(best_x[i])) for i,name in enumerate(PARAM_NAMES)},
                **{f"ga_{name}":float(np.exp(final_x[i])) for i,name in enumerate(PARAM_NAMES)})


def run_ga_multistart(a,y,theta_truth,sigma_truth,true_velocity,freq,seed):
    start=time.perf_counter()
    sample_means=np.asarray(y,dtype=float).mean(axis=0)
    inner_rng=np.random.default_rng(seed)
    trials=[]
    for index in range(a.ga_runs):
        ga_seed=int(inner_rng.integers(0,2**31-1))
        trial=one_ga_start(a,freq,sample_means,ga_seed)
        trial["ga_run"]=index+1
        trials.append(trial)
    valid=[t for t in trials if np.isfinite(t["ga_sse_mean_final"])]
    if not valid:
        raise RuntimeError("GA multistart produced no finite objective")
    best=min(valid,key=lambda t:t["ga_sse_mean_final"])
    theta_hat=np.array([best[f"ga_{name}"] for name in PARAM_NAMES])
    v_hat=ga_velocity_numpy(freq,np.log(theta_hat))
    residuals=np.asarray(y)-v_hat[None,:]
    sse_full=float(np.sum(residuals**2))
    sigma_hat=float(np.sqrt(sse_full/residuals.size))
    ga_x=np.array([[np.log10(trial[f"ga_{name}"]) for name in PARAM_NAMES]
                   for trial in valid])
    # GA-run dispersion represents optimizer multistart variability, NOT a confidence interval.
    spread=float(np.max(np.std(ga_x,axis=0))) if len(valid)>1 else float("nan")
    perparam=[]
    for i,name in enumerate(PARAM_NAMES):
        log_err=float(abs(np.log10(theta_hat[i]/theta_truth[i])))
        perparam.append(dict(parameter=name,true_value=float(theta_truth[i]),
            ga_estimate=float(theta_hat[i]),ga_abs_error_log10=log_err,
            ga_within_factor2=bool(log_err<=math.log10(2)),
            ga_start_std_log10=float(np.std(ga_x[:,i],ddof=1)) if len(valid)>1 else float("nan")))
    e_sigma=float(abs(np.log10(sigma_hat/sigma_truth))) if sigma_hat>0 else float("inf")
    perparam.append(dict(parameter="sigma_y",true_value=float(sigma_truth),
        ga_estimate=sigma_hat,ga_abs_error_log10=e_sigma,
        ga_within_factor2=bool(e_sigma<=math.log10(2)),
        ga_start_std_log10=float("nan")))
    d=dict(ga_status="ok",ga_runs_attempted=len(trials),ga_runs_finite=len(valid),
           ga_best_run=int(best["ga_run"]),ga_best_sse_total=sse_full,
           ga_best_sse_on_means=float(best["ga_sse_mean_final"]),
           ga_sigma_mle=sigma_hat,ga_seconds=time.perf_counter()-start,
           ga_max_start_std_log10=spread,
           ga_velocity_rmse_to_true=float(np.sqrt(np.mean((v_hat-true_velocity)**2))),
           ga_velocity_rmse_to_observations=float(np.sqrt(np.mean(residuals**2))),
           ga_mean_abs_param_log10_error=float(np.mean([r["ga_abs_error_log10"] for r in perparam[:4]])),
           ga_all_4_factor2=bool(all(r["ga_within_factor2"] for r in perparam[:4])),
           **{f"ga_best_{name}":float(theta_hat[i]) for i,name in enumerate(PARAM_NAMES)})
    return d,perparam,trials


def screen_and_select(a, freqs, outdir, pred, jac_batch):
    print(f"[Screen] {a.candidate_draws:,} unrestricted independent log-uniform truths; {len(freqs)} frequencies")
    rng = np.random.default_rng(a.seed)
    draws_ln = rng.uniform(LN_LOW, LN_HIGH, size=(a.candidate_draws, 4))
    rows=[]
    batch_size=a.batch_size
    for start in range(0, a.candidate_draws, batch_size):
        stop = min(start+batch_size, a.candidate_draws)
        jac = np.asarray(jac_batch(jnp.asarray(draws_ln[start:stop])), dtype=np.float64)
        jw = jac*(math.sqrt(a.n_per_frequency)/a.sigma_y)
        s = np.linalg.svd(jw,compute_uv=False)
        with np.errstate(divide="ignore", invalid="ignore"):
            lcond=2*(np.log10(s[:,0])-np.log10(s[:,-1]))
            lmin=2*np.log10(s[:,-1]); lmax=2*np.log10(s[:,0])
            ld=2*np.sum(np.log(s),axis=1)
            lgeom=2*np.mean(np.log10(s),axis=1)
            ltrace=np.logaddexp.reduce(2*np.log(s),axis=1)/np.log(10)
        for ii in range(stop-start):
            row=dict(candidate_id=start+ii+1, **{name:float(np.exp(draws_ln[start+ii,k]))
                                            for k,name in enumerate(PARAM_NAMES)})
            row.update(log10_fim_cond=float(lcond[ii]), log10_lambda_min=float(lmin[ii]),
                       log10_lambda_max=float(lmax[ii]), fim_logdet=float(ld[ii]),
                       log10_fim_info_geom=float(lgeom[ii]),log10_fim_trace=float(ltrace[ii]),
                       numeric_rank=int(np.sum(s[ii]>s[ii,0]*np.finfo(float).eps*max(jw.shape[1:]))))
            rows.append(row)
        if (stop==a.candidate_draws) or (stop//batch_size)%20==0:
            print(f"  screened {stop:,}/{a.candidate_draws:,}", flush=True)
    pool=pd.DataFrame(rows)
    # Use log10 condition in stratification to avoid overflow in very bad regimes.
    edges,labels=stratum_labels(a.fim_bounds, include_s4=(a.sampling_mode == "prior"))
    pool["stratum"]=[label_bin(float(10**x) if np.isfinite(x) and x<308 else float("inf"), edges, labels)
                    for x in pool.log10_fim_cond]
    safe_csv_save(pool, outdir/"screened_prior.csv")
    counts=pool.stratum.value_counts()
    print("[Screen] Broad-prior regime counts (NOT biological frequencies):")
    for label in [*labels, "S4_extreme_excluded", "invalid"]:
        print(f"  {label}: {counts.get(label,0)}/{len(pool)} = {counts.get(label,0)/len(pool):.2%}")
    count_labels=list(dict.fromkeys([*labels, "S4_extreme_excluded", "invalid"]))
    counts_df = pd.DataFrame([dict(stratum=lab, n_screened=int(counts.get(lab, 0)),
                        screened_fraction=float(counts.get(lab, 0)/len(pool)),
                        selected_for_fitting=(lab in labels))
                        for lab in count_labels])
    safe_csv_save(counts_df, outdir/"screening_counts.csv")
    choose_rng=np.random.default_rng(a.seed+1021)
    selected=[]
    if a.truths_csv is None and a.sampling_mode == "prior":
        # Selection is UNCONDITIONAL on FIM, lambda_min or any recovery outcome.
        # Each row of pool was sampled independently from the same log-uniform
        # dielectric prior. Uniform sampling without replacement preserves its
        # prior-predictive distribution. Unlike legacy S1/S2/S3 design, S4 IS kept.
        sel_idx=choose_rng.choice(pool.index.to_numpy(),size=a.prior_truths,replace=False)
        for j,idx in enumerate(sel_idx,1):
            row=pool.iloc[int(idx)].to_dict()
            if row["stratum"]=="invalid":
                raise RuntimeError("An unconditional prior draw produced non-finite FIM. "
                                   "Retain the draw and investigate numerics; do not filter by FIM.")
            row["truth_id"]=f"PRIOR_T{j:04d}"
            selected.append(row)
    elif a.truths_csv is None:
        for lab in labels:
            candidates=pool[pool.stratum==lab].index.to_numpy()
            if len(candidates)<a.truths_per_stratum:
                raise RuntimeError(
                    f"Stratum {lab} has only {len(candidates)} draws, need {a.truths_per_stratum}. "
                    f"Increase --candidate-draws (currently {a.candidate_draws}) or change bins. "
                    "No other strata are substituted silently.")
            sel_idx=choose_rng.choice(candidates, size=a.truths_per_stratum, replace=False)
            for j,idx in enumerate(sel_idx):
                row=pool.iloc[int(idx)].to_dict()
                row["truth_id"]=f"{lab}_T{j+1:03d}"
                selected.append(row)
    if a.truths_csv is not None:
        original=pd.read_csv(a.truths_csv)
        missing=set(PARAM_NAMES)-set(original.columns)
        if missing:
            raise RuntimeError(f"Provided truths CSV missing columns {missing}")
        if len(original)==0:
            raise RuntimeError("Provided truths CSV is empty")
        selected=[]
        for ix, old in original.iterrows():
            theta=old[list(PARAM_NAMES)].to_numpy(dtype=float)
            if not (np.all(np.isfinite(theta)) and np.all(theta>=LOW) and np.all(theta<=HIGH)):
                raise RuntimeError(f"Invalid dielectric parameter at input truth row {ix}")
            jac_i=np.asarray(jac_batch(jnp.asarray(np.log(theta)[None,:])))[0]
            inf=svd_information(jac_i,a.n_per_frequency,a.sigma_y)
            lab=label_bin(inf["fim_cond"],edges,labels)
            if lab not in labels:
                raise RuntimeError(f"Reused truth row {ix} has an invalid FIM stratum for this sampling mode: {lab}")
            truth_id=(str(old["truth_id"]) if "truth_id" in original else f"{lab}_T{ix+1:03d}")
            if "stratum" in original and str(old["stratum"])!=lab:
                raise RuntimeError(f"Reused truth {truth_id} changed stratum {old['stratum']} -> {lab}")
            selected.append(dict(candidate_id=(old["candidate_id"] if "candidate_id" in original else ix+1),
                 **{name:float(theta[k]) for k,name in enumerate(PARAM_NAMES)},
                 log10_fim_cond=inf["log10_fim_cond"],
                 log10_lambda_min=float(np.log10(inf["fim_lmin"])) if inf["fim_lmin"]>0 else -np.inf,
                 log10_lambda_max=float(np.log10(inf["fim_lmax"])) if inf["fim_lmax"]>0 else -np.inf,
                 fim_logdet=inf["fim_logdet"],
                 log10_fim_info_geom=inf["log10_fim_info_geom"],
                 log10_fim_trace=inf["log10_fim_trace"],numeric_rank=inf["rank_numeric"],
                 stratum=lab,truth_id=truth_id))
        if len({x["truth_id"] for x in selected})!=len(selected):
            raise RuntimeError("Duplicate truth IDs in --truths-csv")
        print(f"[Reused truths] {len(selected)} truths from {a.truths_csv}, FIM recomputed at sigma_reference={a.sigma_y}")
    truths=pd.DataFrame(selected)
    # Pre-declare low/mid/high INFORMATION groups by selected-truth terciles of
    # log10(lambda_min) at the same reference sigma as FIM screening.  These
    # DATA-RELATIVE groups are not universal FIM thresholds and are not chosen
    # using GA/Bayesian recovery errors.  Actual-noise groups use these SAME
    # stored numeric boundaries, allowing noise scenarios to change category.
    vals=truths["log10_lambda_min"].to_numpy(dtype=float)
    if not np.all(np.isfinite(vals)):
        raise RuntimeError("Cannot construct information classes: non-finite lambda_min")
    q1,q2=(float(v) for v in np.quantile(vals,[1/3,2/3]))
    if q1>=q2:
        raise RuntimeError("Information terciles are degenerate: select additional distinct truths")
    def _refinfo(v):
        return label_information(float(v),(q1,q2))
    truths["info_class_reference"]=[_refinfo(v) for v in vals]
    # Secondary, overall information measure (geometric mean of 4 eigenvalues)
    geom_q1,geom_q2=(float(v) for v in np.quantile(
        truths["log10_fim_info_geom"].to_numpy(dtype=float),[1/3,2/3]))
    if geom_q1>=geom_q2:
        raise RuntimeError("Overall information terciles are degenerate")
    truths["info_geom_class_reference"]=[label_information(float(v),(geom_q1,geom_q2))
                                               for v in truths.log10_fim_info_geom]
    truths["log10_info_geom_ref_q33"]=geom_q1
    truths["log10_info_geom_ref_q67"]=geom_q2
    truths["log10_information_ref_q33"]=q1
    truths["log10_information_ref_q67"]=q2
    (outdir/"information_reference_bins.json").write_text(json.dumps({
        "primary_information_metric":"log10(lambda_min(F))",
        "description":"Synthetic-selected-truth terciles at fixed sigma reference; not biological or universal thresholds",
        "reference_sigma_um_per_s":float(a.sigma_y),
        "n_per_frequency":int(a.n_per_frequency),
        "log10_lambda_min_lower_cutoff":q1,
        "log10_lambda_min_upper_cutoff":q2,
        "log10_geom_mean_lower_cutoff":geom_q1,
        "log10_geom_mean_upper_cutoff":geom_q2,
        "labels":["I1_low","I2_medium","I3_high"],
        "classification_rule":"I1: x < q33; I2: q33 <= x < q67; I3: x >= q67"
    },indent=2),encoding="utf8")
    safe_csv_save(truths,outdir/"selected_truths.csv")
    print(f"[Selection: {a.sampling_mode}] {len(truths)} independent truths:")
    for lab in labels:
        print(f"  {lab}: {int((truths.stratum==lab).sum())} truths; "
              f"{int((truths.stratum==lab).sum()) * a.datasets_per_truth * len(a.noise_multipliers)} datasets")
    if a.sampling_mode == "stratified":
        print("  S4_extreme_excluded: 0 selected (screened candidates retained)")
    safe_csv_save(pd.DataFrame([dict(stratum=lab, selected_truths=int((truths.stratum==lab).sum()),
                     datasets_planned=int((truths.stratum==lab).sum())*a.datasets_per_truth*len(a.noise_multipliers))
                     for lab in labels]),outdir/"controlled_selection_counts.csv")
    # Store genuine 4x4 Fisher matrices for every selected truth.
    all_f=[]; all_eigs=[]
    for _,row in truths.iterrows():
        lnth=np.log(row[list(PARAM_NAMES)].to_numpy(dtype=float))
        one_jac=np.asarray(jac_batch(jnp.asarray(lnth[None,:])))[0]
        info=svd_information(one_jac,a.n_per_frequency,a.sigma_y)
        all_f.append(info["F"]); all_eigs.append(info["eigs"])
    np.savez_compressed(outdir/"selected_FIM_matrices.npz",
                        truth_ids=truths.truth_id.to_numpy(),
                        F=np.asarray(all_f), eigs=np.asarray(all_eigs),
                        parameter_names=np.asarray(LOG_NAMES))
    return pool,truths,edges,labels


def make_model(freqs, n_per_frequency):
    import numpyro
    import numpyro.distributions as dist
    f=jnp.asarray(freqs)
    lo=jnp.asarray(LN_LOW)
    hi=jnp.asarray(LN_HIGH)
    def model(y):
        ln_params=[]
        for i,n in enumerate(LOG_NAMES):
            ln_params.append(numpyro.sample(n,dist.Uniform(lo[i],hi[i])))
        theta=jnp.exp(jnp.stack(ln_params))
        sigma=numpyro.sample("sigma_y",dist.HalfCauchy(5.0))
        mu=single_velocity(f,theta)
        with numpyro.plate("individual_observations",n_per_frequency):
            numpyro.sample("obs",dist.Normal(mu,sigma).to_event(1),obs=y)
    return model


def normalized_interval_score(low, high, truth, prior_log_width, alpha=.05):
    """Proper central interval score normalized by a FIXED log10-prior span.

    Lower is better; synthetic truth is used ONLY to EVALUATE an interval.
    Do not use this score to gate real-data fits or tune held-out rules.
    """
    vals=np.asarray([low,high,truth,prior_log_width],dtype=float)
    if not np.all(np.isfinite(vals)) or prior_log_width<=0 or high<low:
        return float('nan')
    return float(((high-low)+(2/alpha)*max(low-truth,0.)+
                  (2/alpha)*max(truth-high,0.))/prior_log_width)


def fim_gaussian_recovery_probability(sd_ln, factor):
    """Local untruncated normal approximation for GA log-parameter uncertainty.

    Equal target as posterior probability centered at GA. Not a calibrated
    confidence probability and may fail near bounds / non-linear regions.
    """
    from scipy.special import ndtr
    sd=np.asarray(sd_ln,dtype=float)
    with np.errstate(divide='ignore',invalid='ignore'):
        x=np.log(float(factor))/sd
        result=2.*ndtr(x)-1.
    result=np.where(sd==0.,1.,np.where(sd>0.,result,np.nan))
    return np.clip(result,0.,1.)


def interval_score_ga_fim(est, sd_ln, truth, prior_span):
    if not (np.isfinite(est) and est>0 and np.isfinite(sd_ln) and sd_ln>=0
            and np.isfinite(truth) and truth>0):
        return float('nan')
    center=math.log10(est)
    halfwidth=Z_95*sd_ln/math.log(10.)
    return normalized_interval_score(center-halfwidth,center+halfwidth,
                                     math.log10(truth),prior_span)



# =============================================================================
# JOINT POSTERIOR SUPPORT: four correlated log dielectric parameters
# =============================================================================
def joint_posterior_from_draws(samples, ga_theta, factor, true_theta=None):
    """Compute *joint* events from ALIGNED NUTS draws, not multiplied marginals.

    Does not read ground truth for probabilities; truth is only an optional
    synthetic validation label. Six posterior correlation coefficients and
    all pairwise joint probabilities quantify parameter dependence.
    """
    if factor <= 1 or not np.isfinite(factor):
        raise ValueError('factor must be finite and >1')
    cols=[]
    for key in LOG_NAMES:
        v=np.asarray(samples[key],float).reshape(-1)
        if not np.all(np.isfinite(v)):
            raise ValueError('Nonfinite NUTS log parameter draws: '+key)
        cols.append(v)
    x=np.stack(cols,axis=1)
    if x.ndim!=2 or x.shape[1]!=4 or x.shape[0]<2:
        raise ValueError('Expected aligned Mx4 NUTS posterior log draws')
    ga=np.asarray(ga_theta,float)
    if ga.shape!=(4,) or not np.all(np.isfinite(ga)) or np.any(ga<=0):
        raise ValueError('Valid positive 4D GA estimates required')
    centers=np.median(x,axis=0)
    tau=np.log(factor)
    masks_ga=np.abs(x-np.log(ga)[None,:])<=tau
    masks_b=np.abs(x-centers[None,:])<=tau
    pga=masks_ga.mean(axis=0)
    pb=masks_b.mean(axis=0)
    pair_names={}
    for i in range(4):
        for j in range(i+1,4):
            key=PARAM_NAMES[i]+'__'+PARAM_NAMES[j]
            pair_names['posterior_corr_ln_'+key]=float(np.corrcoef(x[:,i],x[:,j])[0,1])
            pair_names['posterior_pair_GA_'+key]=float(np.mean(masks_ga[:,i]&masks_ga[:,j]))
            pair_names['posterior_pair_Bayes_'+key]=float(np.mean(masks_b[:,i]&masks_b[:,j]))
    ret=dict(posterior_joint_target_factor_used=float(factor),
        posterior_joint_prob_recovery_GA_center=float(masks_ga.all(axis=1).mean()),
        posterior_joint_prob_recovery_Bayes_center=float(masks_b.all(axis=1).mean()),
        posterior_joint_lower_frechet_GA_center=float(max(0.,pga.sum()-3)),
        posterior_joint_upper_frechet_GA_center=float(np.min(pga)),
        posterior_joint_lower_frechet_Bayes_center=float(max(0.,pb.sum()-3)),
        posterior_joint_upper_frechet_Bayes_center=float(np.min(pb)),
        posterior_expected_n_GA_supported=float(pga.sum()),
        posterior_expected_n_Bayes_supported=float(pb.sum()),
        posterior_prob_at_least_3_GA_center=float(np.mean(masks_ga.sum(axis=1)>=3)),
        posterior_prob_at_least_3_Bayes_center=float(np.mean(masks_b.sum(axis=1)>=3)),
        posterior_draws_joint_used=int(x.shape[0]),**pair_names)
    if true_theta is not None:
        true=np.asarray(true_theta,float)
        if true.shape!=(4,) or not np.all(np.isfinite(true)) or np.any(true<=0):
            raise ValueError('Invalid synthetic 4D truth')
        ret.update(ga_joint_recovery_success=bool(np.all(np.abs(np.log(ga/true))<=tau)),
                   bayes_joint_recovery_success=bool(np.all(np.abs(centers-np.log(true))<=tau)))
    return ret


def run_one_mcmc(a, y, true_theta, sigma_true, velocity_true, frequencies, rng_seed, outdir, dataset_id, ga_theta=None):
    import numpyro
    import arviz as az
    from numpyro.infer import MCMC, NUTS
    if a.chain_method == "parallel" and jax.local_device_count() < a.chains:
        raise RuntimeError(
            "Parallel chains require >= --chains visible devices BEFORE JAX initialization. "
            "Use --chain-method sequential (recommended) or launch Python with "
            "XLA_FLAGS=--xla_force_host_platform_device_count=4 for four CPU chains."
        )
    mcmc=MCMC(NUTS(make_model(frequencies,a.n_per_frequency),
                   target_accept_prob=a.target_accept, max_tree_depth=a.max_tree_depth,
                   dense_mass=a.dense_mass),
              num_warmup=a.warmup,num_samples=a.samples,num_chains=a.chains,
              chain_method=a.chain_method, progress_bar=False)
    start=time.perf_counter()
    mcmc.run(jax.random.PRNGKey(rng_seed),y=jnp.asarray(y),
             extra_fields=("diverging","num_steps","accept_prob"))
    seconds=time.perf_counter()-start
    extra=mcmc.get_extra_fields()
    n_div=int(np.asarray(extra.get("diverging",[])).sum())
    total=int(a.chains*a.samples)
    num_steps=np.asarray(extra.get("num_steps",[]))
    max_nodes=2**a.max_tree_depth-1
    depth_hit_rate=float(np.mean(num_steps>=max_nodes)) if num_steps.size else float("nan")
    summary=az.summary(az.from_numpyro(mcmc),var_names=list(LOG_NAMES)+["sigma_y"])
    rhat=np.asarray(summary["r_hat"],float)
    essb=np.asarray(summary["ess_bulk"],float)
    esst=np.asarray(summary["ess_tail"],float)
    rmax=float(np.nanmax(rhat)) if np.isfinite(rhat).any() else float("nan")
    ebmin=float(np.nanmin(essb)) if np.isfinite(essb).any() else float("nan")
    etmin=float(np.nanmin(esst)) if np.isfinite(esst).any() else float("nan")
    passed=bool(n_div==0 and np.all(np.isfinite(rhat)) and np.all(rhat<a.max_rhat)
                and np.all(np.isfinite(essb)) and np.all(essb>a.min_ess)
                and np.all(np.isfinite(esst)) and np.all(esst>a.min_ess))
    samples={k:np.asarray(v) for k,v in mcmc.get_samples().items()}
    # A small deterministic subset of samples is sufficient to render latent
    # predictive bands without saving every NUTS draw for every dataset.
    if a.spectra_plots != "none":
        try:
            full_count=len(samples[LOG_NAMES[0]])
            indices=np.linspace(0,full_count-1,min(a.posterior_curve_draws,full_count),dtype=int)
            log_theta_draws=np.column_stack([samples[key][indices] for key in LOG_NAMES])
            dense_freq=np.geomspace(float(min(frequencies)),float(max(frequencies)),240)
            dense_pred=ga_velocity_numpy(dense_freq,log_theta_draws)
            qlo,qmed,qhi=np.quantile(dense_pred,[0.025,0.5,0.975],axis=0)
            curve_dir=outdir/"posterior_curve_summaries"
            curve_dir.mkdir(exist_ok=True)
            np.savez_compressed(curve_dir/(dataset_id+".npz"),
                frequencies_hz=dense_freq,latent_q025=qlo,latent_q50=qmed,
                latent_q975=qhi,latent_mean=np.mean(dense_pred,axis=0))
        except Exception as exc:
            print(f"  [Posterior latent band warning] {dataset_id}: {exc}",flush=True)
    if a.save_posteriors:
        postdir=outdir/"posterior_samples";postdir.mkdir(exist_ok=True)
        np.savez_compressed(postdir/(dataset_id+".npz"),**samples)
    if a.plot_representatives and dataset_id.endswith("_T001_D001_L01"):
        try:
            chain_samples={k:np.asarray(v) for k,v in mcmc.get_samples(group_by_chain=True).items()}
            labels=list(PARAM_NAMES)+["sigma_y"]
            truth_vals=list(true_theta)+[sigma_true]
            fig,axs=plt.subplots(2,5,figsize=(19,7))
            for idx,label in enumerate(labels):
                key=LOG_NAMES[idx] if idx<4 else "sigma_y"
                arr=chain_samples[key]/math.log(10) if idx<4 else np.log10(chain_samples[key])
                truth_log=math.log10(truth_vals[idx])
                axs[0,idx].hist(arr.ravel(),bins=40,density=True)
                axs[0,idx].axvline(truth_log,ls="--",color="black")
                axs[0,idx].set_title(label)
                axs[0,idx].set_xlabel("log10(parameter)")
                for ci,trace in enumerate(arr):
                    axs[1,idx].plot(trace,alpha=.65,lw=.55,label=f"Chain {ci+1}")
                axs[1,idx].axhline(truth_log,ls="--",color="black",lw=1)
                axs[1,idx].set_xlabel("Posterior iteration")
            axs[1,0].legend(loc="best",fontsize=7)
            fig.suptitle(f"Representative posterior and NUTS traces: {dataset_id}")
            fig.tight_layout()
            pltdir=outdir/"plots";pltdir.mkdir(exist_ok=True)
            fig.savefig(pltdir/("representative_"+dataset_id+".png"),dpi=140)
            plt.close(fig)
        except Exception as exc:
            print(f"[Representative plot skipped] {exc}",flush=True)
    # Descriptive, in-sample PPC using joint posterior draws; this is NOT
    # held-out predictive validation and must not be used as proof of truth recovery.
    try:
        rng_ppc=np.random.default_rng(rng_seed+3491)
        n_avail=len(samples[LOG_NAMES[0]])
        draw_ids=rng_ppc.choice(n_avail,size=min(a.ppc_draws,n_avail),replace=False)
        log_draws=np.column_stack([samples[key][draw_ids] for key in LOG_NAMES])
        vdraw=ga_velocity_numpy(frequencies,log_draws)  # [posterior draws, 6]
        sigdraw=np.asarray(samples['sigma_y'],float)[draw_ids]
        replicates=vdraw+rng_ppc.normal(size=vdraw.shape)*sigdraw[:,None]
        pp_lo,pp_hi=np.quantile(replicates,[.025,.975],axis=0)
        yobs=np.asarray(y,float)
        ppc_point_coverage95=float(np.mean((yobs>=pp_lo[None,:])&(yobs<=pp_hi[None,:])))
        # Predict replicated frequency means with measurement-count weighting.
        mean_replicates=vdraw+rng_ppc.normal(size=vdraw.shape)*sigdraw[:,None]/math.sqrt(a.n_per_frequency)
        mean_lo,mean_hi=np.quantile(mean_replicates,[.025,.975],axis=0)
        ppc_mean_coverage95=float(np.mean((yobs.mean(axis=0)>=mean_lo)&(yobs.mean(axis=0)<=mean_hi)))
        ppc_individual_interval_width_median=float(np.median(pp_hi-pp_lo))
    except Exception as exc:
        print(f'[PPC warning] {dataset_id}: {type(exc).__name__}: {exc}',flush=True)
        ppc_point_coverage95=np.nan
        ppc_mean_coverage95=np.nan
        ppc_individual_interval_width_median=np.nan
    med_theta=[]; param_rows=[]
    if ga_theta is not None:
        ga_theta=np.asarray(ga_theta,dtype=float)
        if ga_theta.shape!=(4,) or not np.all(np.isfinite(ga_theta)) or np.any(ga_theta<=0):
            raise ValueError("Need 4 valid GA estimates to compute GA-centered posterior probabilities")
    for i,name in enumerate(PARAM_NAMES):
        x=np.asarray(samples[LOG_NAMES[i]],float)/math.log(10)
        qs=np.quantile(x,[0.025,0.5,0.975])
        truth_log10=math.log10(true_theta[i])
        e=abs(float(qs[1])-truth_log10)
        med_theta.append(10**qs[1])
        qprecision=float(np.mean(np.abs(x-float(qs[1]))<=math.log10(2)))
        target_radius=math.log10(a.audit_target_factor)
        ga_log10=float(np.log10(ga_theta[i])) if ga_theta is not None else np.nan
        # ONLY posterior samples and the observed-data estimates are used to
        # calculate predictive recovery probabilities (truth is not involved).
        prob_bayes=float(np.mean(np.abs(x-float(qs[1]))<=target_radius))
        prob_ga=(float(np.mean(np.abs(x-ga_log10)<=target_radius))
                 if np.isfinite(ga_log10) else np.nan)
        low_prior,high_prior=math.log10(LOW[i]),math.log10(HIGH[i])
        prob_prior_ga=(min(high_prior,ga_log10+target_radius)-
                       max(low_prior,ga_log10-target_radius))/PRIOR_SPAN_LOG10[i] if np.isfinite(ga_log10) else np.nan
        prob_prior_ga=float(max(0.,min(1.,prob_prior_ga))) if np.isfinite(prob_prior_ga) else np.nan
        lo_prior_c=low_prior+.025*PRIOR_SPAN_LOG10[i]
        hi_prior_c=high_prior-.025*PRIOR_SPAN_LOG10[i]
        param_rows.append(dict(parameter=name,true_value=true_theta[i],
            posterior_target_factor_used=float(a.audit_target_factor),
            posterior_prob_recovery_GA_center=prob_ga,
            posterior_prob_recovery_Bayes_center=prob_bayes,
            prior_prob_target_GA_center=prob_prior_ga,
            posterior_probability_gain_GA_center=(prob_ga/prob_prior_ga if prob_prior_ga>0 else np.nan),
            normalized_interval_score_Bayes95=normalized_interval_score(
                float(qs[0]),float(qs[2]),truth_log10,float(PRIOR_SPAN_LOG10[i])),
            normalized_interval_score_prior95=normalized_interval_score(
                lo_prior_c,hi_prior_c,truth_log10,float(PRIOR_SPAN_LOG10[i])),
            log10_truth=truth_log10,log10_post_q025=float(qs[0]),
            log10_post_median=float(qs[1]),log10_post_q975=float(qs[2]),
            post_median=float(10**qs[1]),post_ci_low=float(10**qs[0]),
            post_ci_high=float(10**qs[2]),width_log10=float(qs[2]-qs[0]),
            width_over_prior=float((qs[2]-qs[0])/(math.log10(HIGH[i])-math.log10(LOW[i]))),
            prior_width_log10=float(PRIOR_SPAN_LOG10[i]),
            posterior_ci_width_percent_prior=float(100*(qs[2]-qs[0])/PRIOR_SPAN_LOG10[i]),
            posterior_ci_95_coverage_truth=bool(qs[0]<=truth_log10<=qs[2]),
            error_abs_log10=e,hit95=bool(qs[0]<=truth_log10<=qs[2]),
            median_within_factor2=bool(e<=math.log10(2)),
            post_mass_within_factor2_of_median=qprecision,
            post_mass_within_target_of_median=float(np.mean(np.abs(x-float(qs[1]))<=math.log10(a.audit_target_factor)))))
        # V5 IMPORTANT: retain exact q=0.1..0.9 credible-mass interval
        # endpoints/radii ONLINE, without requiring 4 GB+ of raw posterior draws.
        # GA-centered radius is symmetric in log10 around GA but contains q
        # JOINT-CHAIN marginal posterior mass; exact from NUTS draws.
        # Synthetic truth is used ONLY for subsequent scoring.
        for qq in np.arange(0.1,1.0,0.1):
            qtag=f'q{int(round(qq*10))}0'  # q10=10%, q20=20%, ..., q90=90%
            if np.isfinite(ga_log10):
                r=float(np.quantile(np.abs(x-ga_log10),qq))
                param_rows[-1][f'v5_ga_centered_radius_log10_{qtag}']=r
                param_rows[-1][f'v5_ga_centered_truth_cover_{qtag}']=bool(
                    abs(truth_log10-ga_log10)<=r)
            else:
                param_rows[-1][f'v5_ga_centered_radius_log10_{qtag}']=np.nan
                param_rows[-1][f'v5_ga_centered_truth_cover_{qtag}']=np.nan
            interval=np.quantile(x,[(1-qq)/2,(1+qq)/2])
            param_rows[-1][f'v5_equal_tail_lo_log10_{qtag}']=float(interval[0])
            param_rows[-1][f'v5_equal_tail_hi_log10_{qtag}']=float(interval[1])
            param_rows[-1][f'v5_equal_tail_truth_cover_{qtag}']=bool(
                interval[0]<=truth_log10<=interval[1])
    sigma_post=np.asarray(samples["sigma_y"],float)
    qsig=np.quantile(sigma_post,[.025,.5,.975])
    sigma_row=dict(parameter="sigma_y",true_value=sigma_true,
        log10_truth=math.log10(sigma_true),
        log10_post_q025=float(np.log10(qsig[0])),
        log10_post_median=float(np.log10(qsig[1])),
        log10_post_q975=float(np.log10(qsig[2])),
        post_median=float(qsig[1]),post_ci_low=float(qsig[0]),post_ci_high=float(qsig[2]),
        width_log10=float(np.log10(qsig[2]/qsig[0])),width_over_prior=float("nan"),
        error_abs_log10=abs(float(np.log10(qsig[1]/sigma_true))),
        hit95=bool(qsig[0]<=sigma_true<=qsig[2]),
        median_within_factor2=bool(abs(np.log10(qsig[1]/sigma_true))<=math.log10(2)),
        post_mass_within_factor2_of_median=float(np.mean(np.abs(np.log10(sigma_post/qsig[1]))<=math.log10(2))))
    param_rows.append(sigma_row)
    vel_pred=np.asarray(single_velocity(jnp.asarray(frequencies),jnp.asarray(med_theta)))
    velocity_rmse=float(np.sqrt(np.mean((vel_pred-velocity_true)**2)))
    noise_rmse=float(np.sqrt(np.mean((vel_pred[None,:]-y)**2)))
    data_row=dict(status="ok",mcmc_pass=passed,
                  ppc_point_coverage95_insample=ppc_point_coverage95,
                  ppc_frequency_mean_coverage95_insample=ppc_mean_coverage95,
                  ppc_individual_interval_width_median=ppc_individual_interval_width_median,
                  n_divergences=n_div,
                  divergence_rate=n_div/total,
                  max_rhat=rmax,min_ess_bulk=ebmin,min_ess_tail=etmin,
                  depth_hit_rate=depth_hit_rate,seconds=seconds,
                  sigma_true=sigma_true,sigma_median=float(qsig[1]),
                  sigma_hit95=bool(qsig[0]<=sigma_true<=qsig[2]),
                  velocity_rmse_to_true=velocity_rmse,
                  velocity_rmse_to_observations=noise_rmse,
                  mean_abs_param_log10_error=float(np.mean([row["error_abs_log10"] for row in param_rows[:4]])),
                  all_4_parameters_hit95=bool(all(row["hit95"] for row in param_rows[:4])),
                  all_4_parameters_factor2=bool(all(row["median_within_factor2"] for row in param_rows[:4])))
    # This is a FOUR-DIMENSIONAL simultaneous event; do not multiply marginals.
    if ga_theta is not None:
        data_row.update(joint_posterior_from_draws(samples,ga_theta,
                        a.audit_target_factor,true_theta=true_theta))
    return data_row,param_rows


def make_summary(dataset_df, param_df,truth_df, labels, a):
    rows=[]
    for lab in labels:
        d=dataset_df[dataset_df.stratum==lab] if not dataset_df.empty else pd.DataFrame()
        p=param_df[(param_df.stratum==lab) & (param_df.parameter!="sigma_y")] if not param_df.empty else pd.DataFrame()
        valid=d[d.status=="ok"] if not d.empty else pd.DataFrame()
        good=valid[valid.mcmc_pass==True] if not valid.empty else pd.DataFrame()
        p_good=p[p.dataset_id.isin(good.dataset_id)] if not p.empty else pd.DataFrame()
        row=dict(stratum=lab,n_truths=int(sum(truth_df.stratum==lab)),
            n_datasets_total=len(d),n_fit_ok=len(valid),n_mcmc_pass=len(good),
            mcmc_pass_rate=float(len(good)/len(d)) if len(d) else float("nan"),
            failure_or_nonconvergence_rate=float(1-len(good)/len(d)) if len(d) else float("nan"),
            median_fim_cond=float(np.median(10**np.asarray(truth_df[truth_df.stratum==lab].log10_fim_cond)))
                            if np.any(truth_df.stratum==lab) else float("nan"),
            median_lambda_min_screen=float(np.median(10**np.asarray(truth_df[truth_df.stratum==lab].log10_lambda_min)))
                            if np.any(truth_df.stratum==lab) else float("nan"),
            median_param_abs_log10_error=float(p_good.error_abs_log10.median()) if len(p_good) else float("nan"),
            median_param_width_log10=float(p_good.width_log10.median()) if len(p_good) else float("nan"),
            pooled_param_95ci_hit_rate=float(p_good.hit95.mean()) if len(p_good) else float("nan"),
            median_velocity_rmse_to_truth=float(good.velocity_rmse_to_true.median()) if len(good) else float("nan"),
            n_independent_truths_passed=int(good.truth_id.nunique()) if len(good) else 0)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_results(screen,truths,data,pars,labels,edges,outdir):
    plots=outdir/"plots";plots.mkdir(exist_ok=True)
    x=screen.log10_fim_cond.to_numpy(dtype=float)
    x=x[np.isfinite(x)]
    if len(x):
        fig,ax=plt.subplots(figsize=(9,5))
        ax.hist(x,bins=90,density=False)
        for b in edges[1:]:ax.axvline(math.log10(b),ls="--",lw=1)
        ax.set(xlabel="log10(actual FIM condition number)",ylabel="Prior draws",
               title="Unrestricted prior FIM landscape (before stratification)")
        fig.tight_layout();fig.savefig(plots/"01_prior_FIM_log_condition.png",dpi=180);plt.close(fig)
    if data.empty:return
    d=data[data.status=="ok"].copy()
    if d.empty:return
    fig,ax=plt.subplots(figsize=(8,4))
    rates=[float(np.mean(data[data.stratum==lab].mcmc_pass==True))
           if len(data[data.stratum==lab]) else np.nan for lab in labels]
    ax.bar(np.arange(len(labels)),rates)
    ax.set_xticks(np.arange(len(labels)),labels,rotation=20,ha="right")
    ax.set_ylim(0,1.04);ax.set_ylabel("Proportion passing MCMC diagnostic gates")
    ax.set_title("Computational reliability by FIM stratum")
    fig.tight_layout();fig.savefig(plots/"02_mcmc_pass_by_stratum.png",dpi=180);plt.close(fig)
    pp=pars[(pars.parameter!="sigma_y") & (pars.dataset_id.isin(d.loc[d.mcmc_pass==True,"dataset_id"]))].copy()
    if pp.empty:return
    fig,ax=plt.subplots(figsize=(8,5))
    for lab in labels:
        sub=pp[pp.stratum==lab]
        if len(sub):ax.scatter(sub.log10_fim_cond,sub.width_log10,label=lab,alpha=.55,s=13)
    ax.set(xlabel="log10(actual FIM condition number of truth)",
           ylabel="95% posterior CI width (log10 parameter)",
           title="Posterior precision vs local conditioning")
    ax.legend();fig.tight_layout();fig.savefig(plots/"03_posterior_width_vs_fim.png",dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5))
    e=[pp.loc[pp.stratum==lab,"error_abs_log10"].to_numpy() for lab in labels]
    ix=[i for i,x in enumerate(e) if len(x)]
    if ix:
        ax.boxplot([e[i] for i in ix],tick_labels=[labels[i] for i in ix],showfliers=True)
    ax.axhline(math.log10(2),ls="--",color="gray",label="Factor-two error")
    ax.set(ylabel="Absolute recovery error (log10 units)",
           title="Posterior median parameter recovery by stratum")
    ax.legend();fig.tight_layout();fig.savefig(plots/"04_param_recovery_by_stratum.png",dpi=180);plt.close(fig)
    hit=np.full((4,len(labels)),np.nan)
    for i,name in enumerate(PARAM_NAMES):
        for j,lab in enumerate(labels):
            subset=pp[(pp.parameter==name)&(pp.stratum==lab)]
            if len(subset):hit[i,j]=subset.hit95.astype(float).mean()
    fig,ax=plt.subplots(figsize=(9,4.5))
    im=ax.imshow(hit,vmin=0,vmax=1,aspect="auto",cmap="viridis")
    ax.set_xticks(np.arange(len(labels)),labels,rotation=20,ha="right")
    ax.set_yticks(np.arange(4),PARAM_NAMES)
    ax.set_title("Conditional 95% credible interval inclusion (descriptive)")
    for i in range(4):
        for j in range(len(labels)):
            if np.isfinite(hit[i,j]):ax.text(j,i,f"{hit[i,j]:.2f}",ha="center",va="center",color="white")
    fig.colorbar(im,ax=ax,label="Fraction containing generating parameter")
    fig.tight_layout();fig.savefig(plots/"05_conditional_CI_inclusion.png",dpi=180);plt.close(fig)


def _load_csv_records(path):
    return pd.read_csv(path).to_dict("records") if path.exists() else []


def _summarize_comparison(ga_d,ga_p,nuts_d,nuts_p,truths,labels,outdir):
    """Produce separate method summaries and exactly paired diagnostic-valid results."""
    gd=ga_d[ga_d.ga_status=="ok"].copy() if not ga_d.empty else pd.DataFrame()
    nd=nuts_d[(nuts_d.status=="ok") & (nuts_d.mcmc_pass==True)].copy() if not nuts_d.empty else pd.DataFrame()
    gp=ga_p[(ga_p.parameter!="sigma_y") & ga_p.dataset_id.isin(gd.dataset_id)].copy() if not ga_p.empty else pd.DataFrame()
    np_=nuts_p[(nuts_p.parameter!="sigma_y") & nuts_p.dataset_id.isin(nd.dataset_id)].copy() if not nuts_p.empty else pd.DataFrame()
    matched_ids=(set(gd.dataset_id)&set(nd.dataset_id)) if not gd.empty and not nd.empty else set()
    pg=gp[gp.dataset_id.isin(matched_ids)] if len(gp) else pd.DataFrame()
    pn=np_[np_.dataset_id.isin(matched_ids)] if len(np_) else pd.DataFrame()
    if not gd.empty and not nd.empty:
        matched_d=gd.merge(nd,on="dataset_id",how="inner",suffixes=("_ga","_nuts"))
        matched_d=matched_d[matched_d.dataset_id.isin(matched_ids)]
    else:
        matched_d=pd.DataFrame()
    if len(pg) and len(pn):
        matched_p=pg.merge(pn,on=["dataset_id","parameter"],how="inner",suffixes=("_ga","_nuts"))
    else:
        matched_p=pd.DataFrame()
    if not matched_d.empty:
        safe_csv_save(matched_d,outdir/"results_paired_datasets.csv")
    if not matched_p.empty:
        safe_csv_save(matched_p,outdir/"results_paired_parameters.csv")
    rows=[]
    for label in labels:
        all_ga=ga_d[ga_d.stratum==label] if not ga_d.empty else pd.DataFrame()
        all_nuts=nuts_d[nuts_d.stratum==label] if not nuts_d.empty else pd.DataFrame()
        ga_ok=gd[gd.stratum==label] if not gd.empty else pd.DataFrame()
        nuts_ok=nd[nd.stratum==label] if not nd.empty else pd.DataFrame()
        p_ga=gp[gp.stratum==label] if len(gp) else pd.DataFrame()
        p_nuts=np_[np_.stratum==label] if len(np_) else pd.DataFrame()
        md=matched_d[matched_d.stratum_ga==label] if len(matched_d) else pd.DataFrame()
        mp=matched_p[matched_p.stratum_ga==label] if len(matched_p) else pd.DataFrame()
        t=truths[truths.stratum==label]
        rows.append(dict(stratum=label,n_independent_truths=len(t),
            n_ga_datasets=len(all_ga),n_ga_success=len(ga_ok),
            ga_success_rate=len(ga_ok)/len(all_ga) if len(all_ga) else float("nan"),
            ga_median_abs_log10_error=float(p_ga.ga_abs_error_log10.median()) if len(p_ga) else float("nan"),
            ga_median_velocity_rmse_to_truth=float(ga_ok.ga_velocity_rmse_to_true.median()) if len(ga_ok) else float("nan"),
            ga_median_runtime_s=float(ga_ok.ga_seconds.median()) if len(ga_ok) else float("nan"),
            n_nuts_datasets=len(all_nuts),n_nuts_diagnostic_pass=len(nuts_ok),
            nuts_pass_rate=len(nuts_ok)/len(all_nuts) if len(all_nuts) else float("nan"),
            nuts_median_abs_log10_error=float(p_nuts.error_abs_log10.median()) if len(p_nuts) else float("nan"),
            nuts_median_width_log10=float(p_nuts.width_log10.median()) if len(p_nuts) else float("nan"),
            nuts_conditional_ci95_truth_inclusion=float(p_nuts.hit95.astype(float).mean()) if len(p_nuts) else float("nan"),
            nuts_median_velocity_rmse_to_truth=float(nuts_ok.velocity_rmse_to_true.median()) if len(nuts_ok) else float("nan"),
            nuts_median_runtime_s=float(nuts_ok.seconds.median()) if len(nuts_ok) else float("nan"),
            n_paired_datasets=len(md),n_paired_parameter_results=len(mp),
            paired_ga_median_abs_log10_error=float(mp.ga_abs_error_log10.median()) if len(mp) else float("nan"),
            paired_nuts_median_abs_log10_error=float(mp.error_abs_log10.median()) if len(mp) else float("nan"),
            n_independent_truths_paired=int(md.truth_id_ga.nunique()) if len(md) else 0))
    ss=pd.DataFrame(rows)
    safe_csv_save(ss,outdir/"summary_GA_NUTS_by_stratum.csv")
    byparam=[]
    for label in labels:
        for pname in (*PARAM_NAMES,"sigma_y"):
            ga_sub=ga_p[(ga_p.stratum==label)&(ga_p.parameter==pname) & ga_p.dataset_id.isin(gd.dataset_id)] if not ga_p.empty and not gd.empty else pd.DataFrame()
            nuts_sub=nuts_p[(nuts_p.stratum==label)&(nuts_p.parameter==pname) & nuts_p.dataset_id.isin(nd.dataset_id)] if not nuts_p.empty and not nd.empty else pd.DataFrame()
            pair_sub=matched_p[(matched_p.stratum_ga==label)&(matched_p.parameter==pname)] if not matched_p.empty else pd.DataFrame()
            byparam.append(dict(stratum=label,parameter=pname,
                n_ga=len(ga_sub),ga_median_abs_log10_error=float(ga_sub.ga_abs_error_log10.median()) if len(ga_sub) else float("nan"),
                ga_factor2_success=float(ga_sub.ga_within_factor2.astype(float).mean()) if len(ga_sub) else float("nan"),
                ga_median_optimizer_sd_log10=float(ga_sub.ga_start_std_log10.median()) if len(ga_sub) and ga_sub.ga_start_std_log10.notna().any() else float("nan"),
                n_nuts_pass=len(nuts_sub),nuts_median_abs_log10_error=float(nuts_sub.error_abs_log10.median()) if len(nuts_sub) else float("nan"),
                nuts_factor2_success=float(nuts_sub.median_within_factor2.astype(float).mean()) if len(nuts_sub) else float("nan"),
                nuts_median_ci_width_log10=float(nuts_sub.width_log10.median()) if len(nuts_sub) else float("nan"),
                nuts_conditional_ci95_truth_inclusion=float(nuts_sub.hit95.astype(float).mean()) if len(nuts_sub) else float("nan"),
                n_paired=len(pair_sub),paired_ga_median_error=float(pair_sub.ga_abs_error_log10.median()) if len(pair_sub) else float("nan"),
                paired_nuts_median_error=float(pair_sub.error_abs_log10.median()) if len(pair_sub) else float("nan")))
    safe_csv_save(pd.DataFrame(byparam),outdir/"summary_GA_NUTS_by_stratum_parameter.csv")
    bytruth=[]
    for _,t in truths.iterrows():
        d_ga=gd[gd.truth_id==t.truth_id] if not gd.empty else pd.DataFrame()
        d_n=nd[nd.truth_id==t.truth_id] if not nd.empty else pd.DataFrame()
        p_pair=matched_p[matched_p.truth_id_ga==t.truth_id] if not matched_p.empty else pd.DataFrame()
        bytruth.append(dict(truth_id=t.truth_id,stratum=t.stratum,
            log10_fim_cond=t.log10_fim_cond,log10_lambda_min=t.log10_lambda_min,
            log10_fim_info_geom_reference=t.log10_fim_info_geom,
            log10_fim_trace_reference=t.log10_fim_trace,
            info_class_reference=t.info_class_reference,
            info_geom_class_reference=t.info_geom_class_reference,
            n_ga_success=len(d_ga),n_nuts_pass=len(d_n),
            n_paired=len(set(d_ga.dataset_id if "dataset_id" in d_ga else []) &
                         set(d_n.dataset_id if "dataset_id" in d_n else [])),
            paired_ga_median_error=float(p_pair.ga_abs_error_log10.median()) if len(p_pair) else float("nan"),
            paired_nuts_median_error=float(p_pair.error_abs_log10.median()) if len(p_pair) else float("nan")))
    safe_csv_save(pd.DataFrame(bytruth),outdir/"summary_GA_NUTS_by_truth.csv")
    return ss,matched_d,matched_p


def _plot_ga_nuts_comparison(matched_d,matched_p,gd,gp,labels,outdir):
    plots=outdir/"plots";plots.mkdir(exist_ok=True)
    if not matched_p.empty:
        fig,ax=plt.subplots(figsize=(10,5))
        positions=[];vals=[];names=[]
        for i,lab in enumerate(labels):
            part=matched_p[(matched_p.stratum_ga==lab)&(matched_p.parameter!="sigma_y")]
            if len(part):
                vals.extend([part.ga_abs_error_log10.to_numpy(),part.error_abs_log10.to_numpy()])
                positions.extend([3*i,3*i+1]);names.extend([lab+" GA",lab+" NUTS"])
        if vals:
            ax.boxplot(vals,positions=positions,widths=.8,showfliers=True)
            ax.set_xticks(positions,names,rotation=30,ha="right")
        ax.axhline(math.log10(2),color="gray",ls="--",label="Factor-two error")
        ax.set_ylabel("Absolute log10 dielectric parameter error")
        ax.set_title("Paired GA vs NUTS: identical datasets, converged NUTS only")
        ax.legend();fig.tight_layout();fig.savefig(plots/"06_paired_GA_vs_NUTS_parameter_error.png",dpi=180);plt.close(fig)
        fig,ax=plt.subplots(figsize=(8,5))
        for lab in labels:
            pp=matched_p[(matched_p.stratum_ga==lab)&(matched_p.parameter!="sigma_y")]
            if len(pp):ax.scatter(pp.ga_abs_error_log10,pp.error_abs_log10,alpha=.5,s=16,label=lab)
        lim=max(float(matched_p.ga_abs_error_log10.max()),float(matched_p.error_abs_log10.max()),.3)
        ax.plot([0,lim],[0,lim],color="gray",ls="--",lw=1)
        ax.set(xlabel="GA absolute log10 parameter error",ylabel="NUTS median absolute log10 error",
               title="Paired parameter recovery: below diagonal favors NUTS")
        ax.legend();fig.tight_layout();fig.savefig(plots/"07_paired_parameter_scatter.png",dpi=180);plt.close(fig)
    if not matched_d.empty:
        fig,ax=plt.subplots(figsize=(8,5))
        for lab in labels:
            dd=matched_d[matched_d.stratum_ga==lab]
            if len(dd):ax.scatter(dd.ga_velocity_rmse_to_true,dd.velocity_rmse_to_true,alpha=.7,s=23,label=lab)
        xmax=max(float(matched_d.ga_velocity_rmse_to_true.max()),float(matched_d.velocity_rmse_to_true.max()),1)
        ax.plot([0,xmax],[0,xmax],ls="--",c="gray",lw=1)
        ax.set(xlabel="GA velocity RMSE to noiseless truth",ylabel="NUTS-median velocity RMSE to truth",
               title="Paired velocity recovery: below diagonal favors NUTS")
        ax.legend();fig.tight_layout();fig.savefig(plots/"08_paired_velocity_RMSE.png",dpi=180);plt.close(fig)
    if not gp.empty:
        fig,ax=plt.subplots(figsize=(8,5))
        for lab in labels:
            gg=gp[gp.stratum==lab]
            if len(gg):ax.scatter(gg.log10_fim_cond,gg.ga_start_std_log10,s=15,alpha=.6,label=lab)
        ax.set(xlabel="log10(FIM condition number)",ylabel="GA start-to-start SD (log10 parameter)",
               title="Multistart optimizer variability (NOT a confidence interval)")
        ax.legend();fig.tight_layout();fig.savefig(plots/"09_GA_multistart_variability.png",dpi=180);plt.close(fig)



# =============================================================================
# MANUSCRIPT-PURPOSE PLOTS: GOOD SPECTRAL FIT DOES NOT GUARANTEE RECOVERY
# =============================================================================
def make_noise_stratum_summary(ga_d,nuts_d,nuts_p,labels,outdir):
    """Avoid treating 3 noise replicates as independent ground truths."""
    if ga_d.empty:
        return
    g=ga_d[ga_d.ga_status=="ok"].copy()
    if g.empty:
        return
    g["ga_latent_rmse_over_noise"]=g.ga_velocity_rmse_to_true/g.sigma_true
    g["ga_observed_rmse_over_noise"]=g.ga_velocity_rmse_to_observations/g.sigma_true
    g["fit_good_params_poor"]=((g.ga_latent_rmse_over_noise<=.5)&
                                  (g.ga_mean_abs_param_log10_error>math.log10(2)))
    safe_csv_save(g,outdir/"results_recovery_vs_fit.csv")
    rows=[]
    for (lab,mult),group in g.groupby(["stratum","noise_multiplier"],sort=True):
        # Truth-specific medians protect against pseudo-replication in comparisons.
        bytruth=group.groupby("truth_id")[
            ["ga_latent_rmse_over_noise","ga_mean_abs_param_log10_error",
             "ga_observed_rmse_over_noise"]].median()
        d=dict(stratum=lab,noise_multiplier=float(mult),
               noise_sd=float(group.sigma_true.iloc[0]),
               n_truths=int(group.truth_id.nunique()),n_datasets=int(len(group)),
               ga_median_truthlevel_latent_rmse_over_noise=float(bytruth.ga_latent_rmse_over_noise.median()),
               ga_median_truthlevel_log10_param_error=float(bytruth.ga_mean_abs_param_log10_error.median()),
               median_log10_lambda_min_at_generating_sigma=float(group.log10_lambda_min_true_sigma.median()),
               ga_good_fit_poor_recovery_n=int(group.fit_good_params_poor.sum()),
               ga_good_fit_poor_recovery_fraction=float(group.fit_good_params_poor.mean()))
        if not nuts_d.empty:
            n=nuts_d[(nuts_d.stratum==lab)&(np.isclose(nuts_d.noise_multiplier,mult))]
            d["nuts_datasets_attempted"]=int(len(n))
            d["nuts_diagnostic_pass_n"]=int((n.mcmc_pass==True).sum()) if "mcmc_pass" in n else 0
            d["nuts_diagnostic_pass_fraction"]=(d["nuts_diagnostic_pass_n"]/len(n)) if len(n) else float("nan")
            if not nuts_p.empty and len(n):
                passed=n.loc[(n.status=="ok")&(n.mcmc_pass==True),"dataset_id"]
                p=nuts_p[nuts_p.dataset_id.isin(passed)&nuts_p.parameter.isin(PARAM_NAMES)]
                d["nuts_passing_median_posterior_CI_width_log10"]=float(p.width_log10.median()) if len(p) else float("nan")
                d["nuts_passing_95ci_truth_inclusion_fraction"]=float(p.hit95.mean()) if len(p) else float("nan")
        rows.append(d)
    safe_csv_save(pd.DataFrame(rows),outdir/"summary_by_stratum_and_noise.csv")


def plot_fit_vs_recovery(ga_d,labels,outdir):
    g=ga_d[ga_d.ga_status=="ok"].copy() if not ga_d.empty else pd.DataFrame()
    if g.empty:
        return
    levels=sorted(g.noise_multiplier.unique())
    fig,axs=plt.subplots(1,len(levels),figsize=(5.4*len(levels),5),squeeze=False,
                         sharex=True,sharey=True)
    cols={lab:c for lab,c in zip(labels,["#2878b5","#ef8a17","#9b59b6","#259e8e"])}
    for ax,level in zip(axs[0],levels):
        subset=g[np.isclose(g.noise_multiplier,level)]
        for lab in labels:
            ss=subset[subset.stratum==lab]
            if len(ss):
                ax.scatter(ss.ga_velocity_rmse_to_true/ss.sigma_true,
                           ss.ga_mean_abs_param_log10_error,color=cols[lab],label=lab,
                           alpha=.73,s=24,edgecolors="none")
        ax.axvline(.5,color="gray",ls="--",lw=1)
        ax.axhline(math.log10(2),color="gray",ls=":",lw=1)
        ax.set_title(f"Generating noise SD = {subset.sigma_true.iloc[0]:.3f} um/s")
        ax.set_xlabel("GA latent velocity RMSE / true noise SD")
        ax.grid(alpha=.15)
    axs[0,0].set_ylabel("GA mean abs. log10 dielectric recovery error")
    axs[0,-1].legend(title="Truth FIM stratum",fontsize=8)
    fig.suptitle("Excellent velocity fit can coexist with poor dielectric recovery",fontsize=13)
    fig.text(.5,.005,"Latent velocity RMSE requires synthetic truth; not available for experimental controls. Dashed thresholds are descriptive.",
             ha="center",fontsize=8)
    fig.tight_layout(rect=[0,.035,1,.94]); fig.savefig(outdir/"plots"/"10_good_spectral_fit_vs_parameter_recovery.png",dpi=190)
    plt.close(fig)


def plot_spectrum_comparison(dataset_id,spectra,ga_d,ga_p,nuts_d,nuts_p,truths,outdir):
    if dataset_id not in set(spectra.dataset_id):
        return
    row=ga_d[ga_d.dataset_id==dataset_id]
    if row.empty or row.iloc[0].ga_status!="ok":
        return
    r=row.iloc[0]
    trow=truths[truths.truth_id==r.truth_id]
    if trow.empty:
        return
    theta_truth=trow.iloc[0][list(PARAM_NAMES)].to_numpy(dtype=float)
    theta_ga=np.array([r[f"ga_best_{name}"] for name in PARAM_NAMES],dtype=float)
    pts=spectra[spectra.dataset_id==dataset_id].sort_values("frequency_hz")
    f=pts.frequency_hz.to_numpy(dtype=float)
    dense=np.geomspace(f.min(),f.max(),240)
    v_truth=ga_velocity_numpy(dense,np.log(theta_truth))
    v_ga=ga_velocity_numpy(dense,np.log(theta_ga))
    nd=nuts_d[nuts_d.dataset_id==dataset_id] if not nuts_d.empty else pd.DataFrame()
    np_rows=nuts_p[(nuts_p.dataset_id==dataset_id)&nuts_p.parameter.isin(PARAM_NAMES)] if not nuts_p.empty else pd.DataFrame()
    theta_nuts=None
    if len(np_rows)==4:
        np_index=np_rows.set_index("parameter")
        theta_nuts=np.array([np_index.loc[name,"post_median"] for name in PARAM_NAMES],dtype=float)
    fig,(ax,ax2)=plt.subplots(1,2,figsize=(13.8,5.3),gridspec_kw={"width_ratios":[1.7,1]})
    ax.plot(dense/1e6,v_truth,color="black",lw=2.1,label="Synthetic truth (noiseless)")
    ax.plot(dense/1e6,v_ga,color="#2268ad",lw=1.9,label="Best GA fit")
    pass_status=None
    if theta_nuts is not None:
        pass_status=bool(nd.iloc[0].mcmc_pass) if len(nd) else None
        ax.plot(dense/1e6,ga_velocity_numpy(dense,np.log(theta_nuts)),
                color="#d56b22",lw=1.7,ls="-" if pass_status else "--",
                label="NUTS marginal-median curve"+(" (diagnostics fail)" if not pass_status else ""))
        curve_file=outdir/"posterior_curve_summaries"/(dataset_id+".npz")
        if curve_file.is_file() and pass_status:
            with np.load(curve_file) as d:
                ax.fill_between(d["frequencies_hz"]/1e6,d["latent_q025"],d["latent_q975"],
                                alpha=.18,color="#d56b22",label="95% posterior latent band")
    ax.errorbar(f/1e6,pts.observed_mean.to_numpy(),yerr=pts.observed_sd.to_numpy(),
                fmt="o",ms=4.5,capsize=2.8,lw=1,color="#377f40",
                label="Observed mean +/- individual SD")
    ax.set_xscale("log"); ax.set_xlabel("DEP frequency (MHz)");ax.set_ylabel("DEP velocity (um/s)")
    ax.set_xticks(f/1e6);ax.set_xticklabels([f"{q/1e6:g}" for q in f]);ax.grid(alpha=.15)
    ax.legend(loc="best",fontsize=8)
    truth_label=["eps_cyto","sigma_cyto","eps_shell","sigma_shell"]
    xpos=np.arange(4)
    ga_err=np.log10(theta_ga/theta_truth)
    if theta_nuts is None:
        ax2.bar(xpos,ga_err,width=.7,color="#2268ad",label="GA")
    else:
        nuts_err=np.log10(theta_nuts/theta_truth)
        ax2.bar(xpos-.18,ga_err,width=.35,color="#2268ad",label="GA")
        ax2.bar(xpos+.18,nuts_err,width=.35,color="#d56b22",label="NUTS median")
    ax2.axhline(0,color="black",lw=.9)
    ax2.axhline(math.log10(2),color="gray",ls="--",lw=1)
    ax2.axhline(-math.log10(2),color="gray",ls="--",lw=1)
    ax2.set_xticks(xpos,truth_label,rotation=30,ha="right")
    ax2.set_ylabel("Signed log10(estimated / true parameter)")
    ax2.set_title("Dielectric parameter recovery (truth known)")
    ax2.grid(axis="y",alpha=.15);ax2.legend(fontsize=8)
    rmse_obs=float(r.ga_velocity_rmse_to_observations)
    rmse_latent=float(r.ga_velocity_rmse_to_true)
    rerr=float(r.ga_mean_abs_param_log10_error)
    fig.suptitle(f"{dataset_id} | log10 kappa={float(r.log10_fim_cond):.2f} | "
                 f"log10 lambda_min={float(r.log10_lambda_min_true_sigma):.2f} ({r.info_class_at_true_sigma}) | "
                 f"GA observed RMSE={rmse_obs:.2f}, latent RMSE={rmse_latent:.2f} um/s | "
                 f"GA parameter MAE={rerr:.3f} log10",fontsize=10)
    fig.text(.5,.012,"Lines at unsampled frequencies are forward-model visualizations; only six marked frequencies are observed. "+
             "Dashed bar thresholds = factor two. NUTS median-parameter curve is not the posterior predictive mean.",
             ha="center",fontsize=8)
    fig.tight_layout(rect=[0,.045,1,.92])
    d=outdir/"plots"/"spectra_and_recovery"
    d.mkdir(exist_ok=True,parents=True)
    fig.savefig(d/(dataset_id+".png"),dpi=175);plt.close(fig)


def make_spectra_plots(a,ga_d,ga_p,nuts_d,nuts_p,truths,outdir):
    if a.spectra_plots=="none" or ga_d.empty:
        return
    path=outdir/"spectra_observed_summary.csv"
    if not path.exists():
        return
    spectra=pd.read_csv(path)
    g=ga_d[ga_d.ga_status=="ok"].copy()
    if g.empty:
        return
    ids=[]; selection=[]
    if a.spectra_plots=="all":
        ids=g.dataset_id.tolist()
    else:
        # Group by local FIM regime AND noise scenario.
        for (lab,mult),sub in g.groupby(["stratum","noise_multiplier"],sort=True):
            sub=sub.copy()
            sub["latent_ratio"]=sub.ga_velocity_rmse_to_true/sub.sigma_true
            viable=sub[sub.latent_ratio<=.5]
            # Highest dielectric error among fits reproducing true velocity to <= 0.5 noise SD.
            paradox=(viable if len(viable) else sub).sort_values(
                ["ga_mean_abs_param_log10_error","latent_ratio"],ascending=[False,True]).iloc[0]
            accurate=sub.sort_values("ga_mean_abs_param_log10_error").iloc[0]
            for tag,r in [("good_fit_poor_recovery",paradox),("accurate_recovery",accurate)]:
                if r.dataset_id not in ids:
                    ids.append(r.dataset_id)
                selection.append(dict(stratum=lab,noise_multiplier=mult,case_type=tag,
                                      dataset_id=r.dataset_id,
                                      ga_latent_rmse_over_noise=float(r.latent_ratio),
                                      ga_mean_abs_param_log10_error=float(r.ga_mean_abs_param_log10_error)))
        safe_csv_save(pd.DataFrame(selection),outdir/"representative_case_selection.csv")
    for did in ids:
        try:
            plot_spectrum_comparison(did,spectra,ga_d,ga_p,nuts_d,nuts_p,truths,outdir)
        except Exception as exc:
            print(f"[Spectra plot warning] {did}: {type(exc).__name__}: {exc}",flush=True)
    print(f"[Spectral figures] {len(ids)} figures in {outdir/'plots'/'spectra_and_recovery'}")


# =============================================================================
# FIM ABSOLUTE INFORMATION x CONDITIONING x RECOVERY (MAIN NEW OUTPUTS)
# =============================================================================
def make_information_recovery_outputs(ga_d,nuts_d,labels,outdir):
    """Save joint FIM/condition/recovery files, including truth-median 3D plots.

    The FIM is the ORACLE expected mean-model FIM at each generating truth and
    generating noise SD, never a GA/Bayesian-data-dependent FIM.  Use 1 point
    per truth/noise level for trend plots (not 3 pseudo-independent replicates).
    """
    if ga_d.empty or "ga_status" not in ga_d:
        return
    g=ga_d[ga_d.ga_status=="ok"].copy()
    if g.empty:
        return
    g["latent_rmse_over_noise"]=g.ga_velocity_rmse_to_true/g.sigma_true
    g["fit_good_recovery_poor"]=((g.latent_rmse_over_noise<=.5)&
         (g.ga_mean_abs_param_log10_error>math.log10(2)))
    pass_n=pd.DataFrame()
    if (not nuts_d.empty and "mcmc_pass" in nuts_d.columns and
            "status" in nuts_d.columns):
        pass_n=nuts_d[(nuts_d.status=="ok")&(nuts_d.mcmc_pass==True)].copy()
    n_small=(pass_n[["dataset_id","mean_abs_param_log10_error",
                     "velocity_rmse_to_true"]].rename(columns={
               "mean_abs_param_log10_error":"nuts_mean_abs_param_log10_error",
               "velocity_rmse_to_true":"nuts_velocity_rmse_to_true"})
             if not pass_n.empty else pd.DataFrame(columns=[
                 "dataset_id","nuts_mean_abs_param_log10_error","nuts_velocity_rmse_to_true"]))
    paired=g.merge(n_small,on="dataset_id",how="left")
    safe_csv_save(paired,outdir/"FIM_condition_information_recovery_all_datasets.csv")
    numeric=["ga_mean_abs_param_log10_error","ga_velocity_rmse_to_true",
             "latent_rmse_over_noise","nuts_mean_abs_param_log10_error",
             "nuts_velocity_rmse_to_true"]
    # group includes both reference class and noise-adjusted class
    idx=["truth_id","stratum","noise_multiplier","sigma_true",
         "log10_fim_cond","log10_lambda_min_true_sigma",
         "log10_fim_info_geom_true_sigma","log10_fim_trace_true_sigma",
         "info_class_reference","info_class_at_true_sigma",
         "info_geom_class_reference","info_geom_class_at_true_sigma"]
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",message="Mean of empty slice",category=RuntimeWarning)
            t=paired.groupby(idx,dropna=False,as_index=False)[numeric].median()
    counts=paired.groupby(["truth_id","noise_multiplier"],as_index=False).agg(
        n_ga_replicates=("dataset_id","count"),
        n_poor_recovery_good_fit=("fit_good_recovery_poor","sum"),
        n_nuts_valid=("nuts_mean_abs_param_log10_error","count"))
    t=t.merge(counts,on=["truth_id","noise_multiplier"],how="left")
    safe_csv_save(t,outdir/"FIM_condition_information_recovery_truth_level.csv")

    # Joint grouping within each noise level.  The 3 information bins are
    # deliberately defined with FIXED reference-sigma numeric cuts (not
    # recomputed from each outcome group).  Empty intersections are expected.
    summary=[]
    for (mult,lab,igroup),sub in t.groupby(
            ["noise_multiplier","stratum","info_class_at_true_sigma"],sort=True):
        finite_n=sub.nuts_mean_abs_param_log10_error.dropna()
        summary.append(dict(noise_multiplier=float(mult),sigma_true_median=float(sub.sigma_true.median()),
            stratum=lab,info_class_at_true_sigma=igroup,
            n_unique_truths=int(sub.truth_id.nunique()),
            median_log10_fim_cond=float(sub.log10_fim_cond.median()),
            median_log10_lambda_min_true_sigma=float(sub.log10_lambda_min_true_sigma.median()),
            median_log10_fim_info_geom_true_sigma=float(sub.log10_fim_info_geom_true_sigma.median()),
            median_GA_log10_recovery_error=float(sub.ga_mean_abs_param_log10_error.median()),
            median_GA_velocity_rmse_over_sigma=float(sub.latent_rmse_over_noise.median()),
            n_GA_good_fit_poor_recovery_replicates=int(sub.n_poor_recovery_good_fit.sum()),
            n_NUTS_truths_with_valid_rep=int(finite_n.size),
            median_NUTS_log10_recovery_error=float(finite_n.median()) if len(finite_n) else float("nan")))
    safe_csv_save(pd.DataFrame(summary),outdir/"summary_condition_information_by_noise.csv")
    summary_geom=[]
    for (mult,lab,igroup),sub in t.groupby(
            ["noise_multiplier","stratum","info_geom_class_at_true_sigma"],sort=True):
        val=sub.nuts_mean_abs_param_log10_error.dropna()
        summary_geom.append(dict(noise_multiplier=float(mult),
            sigma_true_median=float(sub.sigma_true.median()),
            stratum=lab,info_geom_class_at_true_sigma=igroup,
            n_unique_truths=int(sub.truth_id.nunique()),
            median_log10_condition=float(sub.log10_fim_cond.median()),
            median_log10_fim_info_geom=float(sub.log10_fim_info_geom_true_sigma.median()),
            median_ga_log10_recovery_error=float(sub.ga_mean_abs_param_log10_error.median()),
            n_NUTS_truths_with_valid_rep=int(len(val)),
            median_nuts_log10_recovery_error=float(val.median()) if len(val) else float("nan")))
    safe_csv_save(pd.DataFrame(summary_geom),outdir/"summary_condition_geomean_information_by_noise.csv")

    # Truth-level rank correlation: DO NOT interpret conditional stratification
    # as random population prevalence, nor spearman as causal evidence.
    from scipy.stats import spearmanr
    corr=[]
    for mult,part in t.groupby("noise_multiplier"):
        desc=dict(noise_multiplier=float(mult),sigma_true_median=float(part.sigma_true.median()),
                  n_independent_truths=int(part.truth_id.nunique()))
        for suffix,left,right in [
                ("conditioning_vs_weak_info","log10_fim_cond","log10_lambda_min_true_sigma"),
                ("conditioning_vs_overall_info","log10_fim_cond","log10_fim_info_geom_true_sigma"),
                ("weak_vs_overall_info","log10_lambda_min_true_sigma","log10_fim_info_geom_true_sigma")]:
            xx=part[[left,right]].apply(pd.to_numeric,errors="coerce").dropna()
            xx=xx[np.isfinite(xx[left])&np.isfinite(xx[right])]
            desc[f"spearman_{suffix}"]=(float(spearmanr(xx[left],xx[right]).statistic)
                if len(xx)>=4 and xx[left].nunique()>1 and xx[right].nunique()>1 else float("nan"))
        for xname,xcol in [("conditioning","log10_fim_cond"),
                           ("weak_info","log10_lambda_min_true_sigma"),
                           ("overall_info","log10_fim_info_geom_true_sigma")]:
            for yname,ycol in [("GA_recovery","ga_mean_abs_param_log10_error"),
                               ("NUTS_recovery","nuts_mean_abs_param_log10_error")]:
                tmp=part[[xcol,ycol]].apply(pd.to_numeric,errors="coerce").dropna()
                tmp=tmp[np.isfinite(tmp[xcol])&np.isfinite(tmp[ycol])]
                desc[f"spearman_{xname}_vs_{yname}"]=(float(spearmanr(tmp[xcol],tmp[ycol]).statistic)
                    if len(tmp)>=4 and tmp[xcol].nunique()>1 and tmp[ycol].nunique()>1 else float("nan"))
        corr.append(desc)
    safe_csv_save(pd.DataFrame(corr),outdir/"summary_FIM_information_recovery_correlations.csv")

    plots=outdir/"plots";plots.mkdir(exist_ok=True)
    levels=sorted(t.noise_multiplier.unique())
    lab_color={lab:c for lab,c in zip(labels,["#2878b5","#ef8a17","#9b59b6","#259e8e"])}
    # 2D information vs recovery, faceted by generating noise
    fig,axs=plt.subplots(1,len(levels),figsize=(5.3*len(levels),4.9),squeeze=False,sharey=True)
    for ax,lvl in zip(axs[0],levels):
        part=t[np.isclose(t.noise_multiplier,lvl)]
        for lab in labels:
            z=part[part.stratum==lab]
            if len(z):
                ax.scatter(z.log10_lambda_min_true_sigma,z.ga_mean_abs_param_log10_error,
                           alpha=.75,s=39,label=lab,color=lab_color[lab],edgecolors="none")
        ax.axhline(math.log10(2),ls="--",lw=.8,color="gray")
        ax.set(title=f"sigma={part.sigma_true.median():.3g} um/s",
               xlabel="log10 lambda_min(F) at true sigma")
        ax.grid(alpha=.16)
    axs[0,0].set_ylabel("GA mean abs log10 dielectric recovery error")
    axs[0,-1].legend(fontsize=8,title="Condition-number regime")
    fig.suptitle("Weakest-direction FIM information vs GA parameter recovery (truth medians)")
    fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(plots/"11_FIM_information_vs_GA_recovery.png",dpi=175);plt.close(fig)

    # 2D x condition, y information, colour recovery.  Avoid creating a
    # misleading second mapping from one-dimensional kappa thresholds.
    fig,axs=plt.subplots(1,len(levels),figsize=(5.0*len(levels),5.2),squeeze=False,sharex=True,sharey=True)
    hi=max(.30103,float(np.nanquantile(t.ga_mean_abs_param_log10_error,.95)))
    sc=None
    for ax,lvl in zip(axs[0],levels):
        sub=t[np.isclose(t.noise_multiplier,lvl)]
        sc=ax.scatter(sub.log10_fim_cond,sub.log10_lambda_min_true_sigma,
            c=sub.ga_mean_abs_param_log10_error,cmap="viridis",vmin=0,vmax=hi,s=45,alpha=.87)
        ax.set(title=f"sigma={sub.sigma_true.median():.3g} um/s",
               xlabel="log10 condition number")
        ax.grid(alpha=.18)
    axs[0,0].set_ylabel("log10 lambda_min(F) at true sigma")
    fig.suptitle("Conditioning, absolute information and recovery (truth-level medians)")
    fig.subplots_adjust(left=.07,right=.875,top=.87,bottom=.15,wspace=.13)
    cax=fig.add_axes([.902,.23,.012,.57])
    fig.colorbar(sc,cax=cax,label="GA log10 recovery error (95th pct clipped)")
    fig.savefig(plots/"12_FIM_condition_information_recovery_2D.png",dpi=175);plt.close(fig)

    # Cross-classification heatmaps: missing combinations stay blank rather than
    # being assigned zero recovery error.  Counts prevent overinterpreting tiny cells.
    from matplotlib.colors import Normalize
    fig,axes=plt.subplots(1,len(levels),figsize=(4.7*len(levels),4.4),squeeze=False)
    cond_order=list(labels)
    info_order=["I1_low","I2_medium","I3_high"]
    allmed=np.asarray([q["median_GA_log10_recovery_error"] for q in summary],float)
    scale=max(.30103,float(np.nanquantile(allmed,.95)))
    for ax,lvl in zip(axes[0],levels):
        grid=np.full((len(cond_order),3),np.nan)
        counts_grid=np.zeros((len(cond_order),3),dtype=int)
        for i,cond_lab in enumerate(cond_order):
            for j,info_lab in enumerate(info_order):
                v=[q for q in summary if np.isclose(q["noise_multiplier"],lvl) and
                   q["stratum"]==cond_lab and q["info_class_at_true_sigma"]==info_lab]
                if v:
                    grid[i,j]=v[0]["median_GA_log10_recovery_error"]
                    counts_grid[i,j]=v[0]["n_unique_truths"]
        im=ax.imshow(np.ma.masked_invalid(grid),cmap="viridis",vmin=0,vmax=scale,aspect="auto")
        ax.set_xticks(range(3),info_order,rotation=25,ha="right",fontsize=8)
        ax.set_yticks(range(len(cond_order)),cond_order,fontsize=8)
        part=t[np.isclose(t.noise_multiplier,lvl)]
        ax.set_title(f"sigma={part.sigma_true.median():.3g} um/s")
        ax.set_xlabel("Weak-direction information class")
        for i in range(len(cond_order)):
            for j in range(3):
                cell=(f"{grid[i,j]:.2f}\nn={counts_grid[i,j]}" if counts_grid[i,j] else "no truths")
                ax.text(j,i,cell,ha="center",va="center",fontsize=8,
                        color="white" if np.isfinite(grid[i,j]) and grid[i,j]>.5*scale else "black")
    axes[0,0].set_ylabel("FIM condition-number class")
    fig.suptitle("Conditioning x weak information: median GA recovery (unique truths)")
    fig.subplots_adjust(left=.09,right=.86,bottom=.23,top=.86,wspace=.30)
    cax=fig.add_axes([.89,.24,.013,.56])
    fig.colorbar(im,cax=cax,label="Median GA log10 recovery error")
    fig.savefig(plots/"15_FIM_condition_information_classification_GA_heatmap.png",dpi=175)
    plt.close(fig)

    # Requested THREE-DIMENSIONAL diagrams, with strictly the three axes:
    # x=log10 kappa(F), y=log10 lambda_min(F), z=recovery error.
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    def plot_3d(method,column,filename):
        use=t.dropna(subset=[column]).copy()
        if use.empty:return
        zlim=max(.30103,float(np.quantile(use[column],.98))*1.10)
        fig=plt.figure(figsize=(5.7*len(levels),5.65))
        for i,lvl in enumerate(levels,1):
            ax=fig.add_subplot(1,len(levels),i,projection="3d")
            sub=use[np.isclose(use.noise_multiplier,lvl)]
            for lab in labels:
                q=sub[sub.stratum==lab]
                if len(q):
                    ax.scatter(q.log10_fim_cond,q.log10_lambda_min_true_sigma,q[column],
                        s=28,color=lab_color[lab],alpha=.80,label=lab,depthshade=False)
            ax.set(xlabel="log10 kappa(F)",ylabel="log10 lambda_min(F)",
                   zlabel="Mean abs log10 recovery error")
            ax.set_title(f"sigma = {t[np.isclose(t.noise_multiplier,lvl)].sigma_true.median():.3g} um/s\n"
                         f"n = {len(sub)} truths",fontsize=10)
            ax.view_init(elev=21,azim=-64)
            ax.set_zlim(bottom=0,top=zlim)
            ax.tick_params(labelsize=7,pad=1)
            if i==len(levels):ax.legend(loc="upper left",fontsize=7)
        fig.suptitle(method+": FIM condition x absolute information x recovery",fontsize=13)
        fig.subplots_adjust(left=.01,right=.99,bottom=.04,top=.87,wspace=.09)
        fig.savefig(plots/filename,dpi=190);plt.close(fig)
    plot_3d("GA", "ga_mean_abs_param_log10_error",
            "13_3D_FIM_condition_information_GA_recovery.png")
    plot_3d("NUTS (diagnostic-pass only)","nuts_mean_abs_param_log10_error",
            "14_3D_FIM_condition_information_NUTS_recovery.png")
    print(f"[FIM information] {len(t)} truth/noise groups, {len(summary)} occupied 3x3xnoise cells; "
          f"joint CSVs + plots 11-15 saved",flush=True)



# =============================================================================
# THREE EVALUATIONS: RECOVERY ACCURACY / POSTERIOR COVERAGE / PRECISION
# And parameter-specific NOISE-ADJUSTED FIM in ln(theta) coordinates.
# =============================================================================
def _fim_precision_per_parameter(jac, noise_sd, n_per_freq):
    """Stable FULL-inverse FIM diagonal via SVD, not reciprocal diagonal FIM.

    For phi=ln(theta):  F = (n/sigma^2) J.T J.
    s_i = sqrt([F^{-1}]_ii); 95% Gaussian half-width=Z_95*s_i.
    For near-singular cases, huge values are kept rather than prior-clipped.
    """
    jac=np.asarray(jac,dtype=float)
    counts=np.asarray(n_per_freq,dtype=float)
    if counts.ndim==0:
        counts=np.full(jac.shape[0],float(counts))
    if counts.shape!=(jac.shape[0],) or np.any(~np.isfinite(counts)) or np.any(counts<=0):
        raise ValueError("n_per_freq must be positive per-frequency counts or a scalar.")
    jw=jac*(np.sqrt(counts)/float(noise_sd))[:,None]
    u,sv,vt=np.linalg.svd(jw,full_matrices=False)
    if sv.size!=4 or not np.all(np.isfinite(sv)) or np.min(sv)<=0:
        return np.full(4,np.nan)
    with np.errstate(over='ignore',divide='ignore',invalid='ignore'):
        return np.sqrt(np.sum((vt.T/sv)**2,axis=1))


def build_three_evaluation_outputs(a,ga_d,ga_p,nuts_d,nuts_p,oracle_widths,jac_batch,outdir):
    """Adds CSVs that use only observable diagnostics (GA-FIM, NUTS posterior),
    with truth-based columns separately labelled for synthetic validation.

    MCMC failures remain in parameter rows with mcmc_pass=False and are
    excluded from aggregate Bayesian coverage and precision summaries.
    """
    if ga_d.empty or ga_p.empty or 'ga_status' not in ga_d:
        return
    gd=ga_d.loc[ga_d.ga_status=='ok'].copy()
    gp=ga_p.loc[ga_p.parameter.isin(PARAM_NAMES) & ga_p.dataset_id.isin(gd.dataset_id)].copy()
    if gd.empty or gp.empty:
        return
    nd=(nuts_d.copy() if not nuts_d.empty else pd.DataFrame())
    np_=(nuts_p.copy() if not nuts_p.empty else pd.DataFrame())
    if not nd.empty and 'mcmc_pass' not in nd:
        nd['mcmc_pass']=False
    if not nd.empty and 'status' in nd:
        nd['mcmc_pass']=(nd.mcmc_pass.astype(str).str.lower()=='true')&(nd.status=='ok')

    # Actual GA-location FIM with *fitted* noise SD: practical without known truth.
    # Batch JAX evaluations to avoid compiling an individual Jacobian per dataset.
    gd=gd.drop_duplicates('dataset_id')
    ids=list(gd.dataset_id)
    estimates=gd[[f'ga_best_{p}' for p in PARAM_NAMES]].to_numpy(dtype=float)
    ln_est=np.log(estimates)
    ga_rows=[]
    for start in range(0,len(ids),a.batch_size):
        end=min(start+a.batch_size,len(ids))
        js=np.asarray(jac_batch(jnp.asarray(ln_est[start:end])),dtype=float)
        for j in range(start,end):
            r=gd.iloc[j]
            sig=float(r.ga_sigma_mle)
            sd=_fim_precision_per_parameter(js[j-start],sig,a.n_per_frequency)
            for i,p in enumerate(PARAM_NAMES):
                w_log10=2*Z_95*sd[i]/math.log(10)
                ga_rows.append(dict(dataset_id=ids[j],parameter=p,
                    ga_fim_sd_ln=sd[i], ga_fim_sd_log10=sd[i]/math.log(10),
                    ga_fim_95_halfwidth_ln=Z_95*sd[i],
                    ga_fim_95_width_log10=w_log10,
                    ga_fim_95_width_percent_prior=100*w_log10/PRIOR_SPAN_LOG10[i],
                    ga_fim_precision_Q_factor2=Z_95*sd[i]/math.log(2)))
    ga_fim=pd.DataFrame(ga_rows)

    # Small, stable core: left side always includes a synthetic GA estimate.
    cols=['dataset_id','truth_id','stratum','noise_multiplier','sigma_true',
          'log10_fim_cond','log10_lambda_min_true_sigma',
          'log10_fim_info_geom_true_sigma','info_class_reference',
          'info_class_at_true_sigma','ga_velocity_rmse_to_true',
          'ga_velocity_rmse_to_observations','ga_sigma_mle','ga_max_start_std_log10']
    context=gd[[c for c in cols if c in gd.columns]].copy()
    p=gp[['dataset_id','parameter','true_value','ga_estimate',
          'ga_abs_error_log10','ga_within_factor2','ga_start_std_log10']].merge(
              context,on='dataset_id',how='left',validate='many_to_one')
    p=p.merge(ga_fim,on=['dataset_id','parameter'],how='left',validate='one_to_one')
    p=p.rename(columns={'ga_abs_error_log10':'ga_recovery_error_log10',
                        'ga_within_factor2':'ga_recovery_within_factor2',
                        'true_value':'synthetic_truth'})
    p['prior_span_log10']=[float(PRIOR_SPAN_LOG10[PARAM_NAMES.index(i)]) for i in p.parameter]
    p['ga_fim_prob_target']=fim_gaussian_recovery_probability(
        pd.to_numeric(p.ga_fim_sd_ln,errors='coerce').to_numpy(float),a.audit_target_factor)
    p['normalized_interval_score_GA_FIM95']=[interval_score_ga_fim(
        float(r.ga_estimate),float(r.ga_fim_sd_ln),float(r.synthetic_truth),float(r.prior_span_log10))
        for r in p.itertuples(index=False)]
    p['ga_recovery_z_oracle']=np.nan
    p['ga_recovery_z_ga_fim']=np.log(10)*p.ga_recovery_error_log10/p.ga_fim_sd_ln
    # Oracle expected FIM evaluated AT TRUE theta and TRUE sigma.
    # It is validation-only and cannot be computed from experimental data.
    usecols=['truth_id','noise_multiplier','parameter','fim_predicted_95_width_log10',
             'fim_width_over_prior','oracle_fim_sd_ln','oracle_fim_95_halfwidth_ln',
             'oracle_fim_precision_Q_factor2']
    oc=oracle_widths[usecols].copy()
    p=p.merge(oc,on=['truth_id','noise_multiplier','parameter'],how='left',validate='many_to_one')
    p['ga_recovery_z_oracle']=np.log(10)*p.ga_recovery_error_log10/p.oracle_fim_sd_ln
    p['oracle_fim_width_percent_prior']=100*p.fim_width_over_prior

    # Posterior CI and diagnostic status. Nonpassing NUTS intervals are
    # present for forensic review, but NEVER counted as validated coverage.
    if not nd.empty:
        dn=nd[['dataset_id','mcmc_pass','status','max_rhat','min_ess_bulk',
               'min_ess_tail','n_divergences','sigma_median']].copy()
        p=p.merge(dn,on='dataset_id',how='left',validate='many_to_one')
    else:
        p['mcmc_pass']=False
    p['mcmc_pass']=p.mcmc_pass.fillna(False).astype(bool)
    if not np_.empty:
        keep=['dataset_id','parameter','post_median','post_ci_low','post_ci_high',
              'width_log10','width_over_prior','error_abs_log10','hit95',
              'median_within_factor2','post_mass_within_factor2_of_median',
               'post_mass_within_target_of_median',
               'posterior_prob_recovery_GA_center','posterior_prob_recovery_Bayes_center',
               'prior_prob_target_GA_center','posterior_probability_gain_GA_center',
               'normalized_interval_score_Bayes95','normalized_interval_score_prior95',
               'posterior_target_factor_used']
        np_small=np_.loc[np_.parameter.isin(PARAM_NAMES),[c for c in keep if c in np_.columns]].copy()
        np_small=np_small.rename(columns={
            'width_log10':'posterior_ci_width_log10',
            'width_over_prior':'posterior_ci_width_fraction_prior',
            'error_abs_log10':'bayes_recovery_error_log10',
            'hit95':'posterior_ci_covers_synthetic_truth',
            'median_within_factor2':'bayes_recovery_within_factor2',
            'post_mass_within_factor2_of_median':'posterior_mass_within_factor2_of_median',
            'post_mass_within_target_of_median':'posterior_mass_within_target_of_median'})
        p=p.merge(np_small,on=['dataset_id','parameter'],how='left',validate='one_to_one')
    for c in ['posterior_ci_width_log10','posterior_ci_width_fraction_prior',
              'bayes_recovery_error_log10','posterior_mass_within_factor2_of_median']:
        if c not in p: p[c]=np.nan
    if 'posterior_ci_covers_synthetic_truth' not in p:
        p['posterior_ci_covers_synthetic_truth']=np.nan
    if 'bayes_recovery_within_factor2' not in p:
        p['bayes_recovery_within_factor2']=np.nan
    # Fields are absent on old results that did not save GA-centered posterior
    # probabilities. Never substitute median-centered probabilities for them.
    for missing_col in ['posterior_prob_recovery_GA_center',
                        'posterior_prob_recovery_Bayes_center','prior_prob_target_GA_center',
                        'posterior_probability_gain_GA_center',
                        'normalized_interval_score_Bayes95','normalized_interval_score_prior95']:
        if missing_col not in p: p[missing_col]=np.nan
    p['posterior_ci_width_percent_prior']=100*p.posterior_ci_width_fraction_prior
    p['posterior_ci_width_remaining_percent_prior']=100-p.posterior_ci_width_percent_prior
    p['ga_bayes_disagreement_log10']=np.where(
        ('post_median' in p.columns) & (pd.to_numeric(p.get('post_median',pd.Series(np.nan,index=p.index)),errors='coerce')>0),
        np.abs(np.log10(p.ga_estimate/p.get('post_median',pd.Series(np.nan,index=p.index)))),np.nan)
    p['bayes_recovery_z_oracle']=np.log(10)*p.bayes_recovery_error_log10/p.oracle_fim_sd_ln
    # Values are NaN when NUTS is missing; never zero-impute coverage or precision.
    for col in ['posterior_ci_covers_synthetic_truth','bayes_recovery_within_factor2']:
        p[col]=p[col].map(lambda x: float(x) if pd.notna(x) else np.nan)
    safe_csv_save(p,outdir/'evaluation_parameter_level.csv')

    # Dataset-level summaries; all 4 dielectric parameters with MCMC pass.
    drows=[]
    for did,part in p.groupby('dataset_id',sort=False):
        first=part.iloc[0]
        passing=bool(first.mcmc_pass)
        have_bayes=passing and part.posterior_ci_width_log10.notna().sum()==4
        drows.append(dict(dataset_id=did,truth_id=first.truth_id,stratum=first.stratum,
            noise_multiplier=float(first.noise_multiplier),sigma_true=float(first.sigma_true),
            log10_fim_cond=float(first.log10_fim_cond),
            log10_lambda_min_true_sigma=float(first.log10_lambda_min_true_sigma),
            ga_velocity_rmse_to_true=float(first.ga_velocity_rmse_to_true),
            ga_sigma_mle=float(first.ga_sigma_mle),mcmc_pass=passing,
            ga_mean_recovery_error_log10=float(part.ga_recovery_error_log10.mean()),
            ga_all4_factor2=bool(part.ga_recovery_within_factor2.all()),
            ga_mean_recovery_z_ga_fim=float(part.ga_recovery_z_ga_fim.mean()),
            ga_median_fim95_percent_prior=float(part.ga_fim_95_width_percent_prior.median()),
            bayes_mean_recovery_error_log10=(float(part.bayes_recovery_error_log10.mean()) if have_bayes else np.nan),
            bayes_all4_factor2=(bool(part.bayes_recovery_within_factor2.astype(bool).all()) if have_bayes else np.nan),
            bayes_fraction_parameters_ci95_cover_truth=(float(part.posterior_ci_covers_synthetic_truth.mean()) if have_bayes else np.nan),
            bayes_all4_ci95_cover_truth=(bool(part.posterior_ci_covers_synthetic_truth.astype(bool).all()) if have_bayes else np.nan),
            bayes_median_ci95_percent_prior=(float(part.posterior_ci_width_percent_prior.median()) if have_bayes else np.nan),
            ga_bayes_median_disagreement_log10=(float(part.ga_bayes_disagreement_log10.median()) if have_bayes else np.nan)))
    d=pd.DataFrame(drows)
    safe_csv_save(d,outdir/'evaluation_dataset_level.csv')

    # Aggregates: group by stratum, noise, parameter. MCMC pass is explicitly
    # the eligibility condition. Truth counts visible to avoid pseudoreplication.
    summ=[]
    for (stratum,mult,param),q in p.groupby(['stratum','noise_multiplier','parameter'],sort=True):
        passed=q.loc[q.mcmc_pass].copy()
        summ.append(dict(stratum=stratum,noise_multiplier=float(mult),parameter=param,
            n_independent_truths=int(q.truth_id.nunique()),n_ga_datasets=int(len(q)),
            n_nuts_diagnostic_pass=int(len(passed)),
            ga_median_recovery_error_log10=float(q.ga_recovery_error_log10.median()),
            ga_factor2_success_rate=float(q.ga_recovery_within_factor2.astype(float).mean()),
            ga_median_fim_sd_ln=float(q.ga_fim_sd_ln.median()),
            ga_median_fim95_width_percent_prior=float(q.ga_fim_95_width_percent_prior.median()),
            oracle_median_fim95_width_percent_prior=float(q.oracle_fim_width_percent_prior.median()),
            bayes_median_recovery_error_log10=(float(passed.bayes_recovery_error_log10.median()) if len(passed) else np.nan),
            bayes_factor2_success_rate=(float(passed.bayes_recovery_within_factor2.mean()) if len(passed) else np.nan),
            bayes_ci95_truth_coverage=(float(passed.posterior_ci_covers_synthetic_truth.mean()) if len(passed) else np.nan),
            bayes_median_ci95_width_log10=(float(passed.posterior_ci_width_log10.median()) if len(passed) else np.nan),
            bayes_median_ci95_width_percent_prior=(float(passed.posterior_ci_width_percent_prior.median()) if len(passed) else np.nan),
            bayes_median_posterior_mass_factor2=(float(passed.posterior_mass_within_factor2_of_median.median()) if len(passed) else np.nan)))
    sm=pd.DataFrame(summ)
    safe_csv_save(sm,outdir/'evaluation_summary_by_stratum_noise_parameter.csv')

    # Aggregate within independent truth FIRST then summarize truth medians;
    # retains conventional observation-level summaries separately above.
    pass_p=p.loc[p.mcmc_pass].copy()
    tcols=['truth_id','stratum','noise_multiplier','parameter']
    if not pass_p.empty:
        truth_avg=pass_p.groupby(tcols,as_index=False).agg(
            ci95_coverage_per_truth=('posterior_ci_covers_synthetic_truth','mean'),
            ci95_width_percent_prior_median=('posterior_ci_width_percent_prior','median'),
            bayes_recovery_error_log10_median=('bayes_recovery_error_log10','median'),
            ga_recovery_error_log10_median=('ga_recovery_error_log10','median'))
        safe_csv_save(truth_avg,outdir/'evaluation_truth_level_passing_NUTS.csv')

    plots=outdir/'plots';plots.mkdir(exist_ok=True)
    # Three evaluation axes, all four parameters, each of 3 noise SD.
    levels=sorted(p.noise_multiplier.dropna().unique())
    fig,axes=plt.subplots(3,len(levels),figsize=(5.2*len(levels),11.2),
                          squeeze=False)
    for col,lvl in enumerate(levels):
        z=p[np.isclose(p.noise_multiplier,lvl)]
        sig=float(z.sigma_true.median())
        ax=axes[0,col]
        ax.scatter(z.log10_fim_cond,z.ga_recovery_error_log10,s=15,alpha=.35,label='GA')
        sub=z[z.mcmc_pass]
        if len(sub):ax.scatter(sub.log10_fim_cond,sub.bayes_recovery_error_log10,s=12,alpha=.3,label='Bayesian median')
        ax.axhline(FACTOR2_LOG10,c='gray',ls='--',lw=1,label='factor 2')
        ax.set_title(f'Generating noise SD = {sig:.3g} um/s')
        ax.grid(alpha=.16)
        if col==0:ax.set_ylabel('1. Recovery |log10(estimate/truth)|')
        if col==len(levels)-1:ax.legend(fontsize=8)
        ax=axes[1,col]
        if len(sub):
            g=sub.groupby('stratum',sort=True).posterior_ci_covers_synthetic_truth.agg(['mean','count'])
            for k,(lab,row) in enumerate(g.iterrows()):
                ax.bar(k,float(row['mean']),width=.7,label=lab)
                ax.text(k,float(row['mean'])+.025,f"n={int(row['count'])}",ha='center',fontsize=8)
            ax.set_xticks(range(len(g)));ax.set_xticklabels(list(g.index),rotation=15)
        ax.axhline(.95,ls='--',c='gray',lw=1,label='nominal 95%')
        ax.set_ylim(0,1.11)
        if not len(sub):
            ax.text(.5,.52,'No diagnostic-valid NUTS results yet',
                    transform=ax.transAxes,ha='center',va='center',fontsize=9,color='gray')
        if col==0:ax.set_ylabel('2. P(truth in 95% CI), passing NUTS')
        ax.grid(alpha=.13,axis='y')
        ax=axes[2,col]
        if len(sub):
            vals=[sub.loc[sub.parameter==pr,'posterior_ci_width_percent_prior'].dropna().values for pr in PARAM_NAMES]
            ax.boxplot(vals,tick_labels=list(PARAM_NAMES),showfliers=False)
        else:
            ax.text(.5,.52,'Posterior CI unavailable (GA only)',
                    transform=ax.transAxes,ha='center',va='center',fontsize=9,color='gray')
            ax.set_xticks(range(1,5));ax.set_xticklabels(PARAM_NAMES)
        ax.set_ylabel('3. CI width / log-prior span (%)' if col==0 else '')
        ax.tick_params(axis='x',rotation=35,labelsize=8)
        ax.set_xlabel('Dielectric parameter')
        ax.grid(alpha=.14,axis='y')
    fig.suptitle('Three evaluations: recovery, CI truth coverage, posterior precision (synthetic)')
    fig.tight_layout(rect=[0,.02,1,.967]);fig.savefig(plots/'16_three_evaluations_recovery_coverage_precision.png',dpi=175);plt.close(fig)

    # Width vs FIM precision: compare correctly in 95% log10 width units;
    # large FIM widths can exceed finite prior widths (not a plotting error).
    if not pass_p.empty:
        fig,axs=plt.subplots(1,4,figsize=(16.5,4.5))
        for ax,param in zip(axs,PARAM_NAMES):
            q=pass_p.loc[pass_p.parameter==param]
            ax.scatter(q.ga_fim_95_width_percent_prior,q.posterior_ci_width_percent_prior,
                c=q.log10_fim_cond,s=14,alpha=.55,cmap='viridis')
            ax.axhline(100,ls=':',c='gray',lw=.8)
            ax.set_title(param)
            ax.set_xlabel('GA-FIM 95% width / log-prior span (%)')
            ax.grid(alpha=.15)
        axs[0].set_ylabel('Bayesian CI95 width / log-prior span (%)')
        fig.suptitle('Local FIM precision vs Bayesian posterior CI precision')
        fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(plots/'17_FIM_vs_posterior_prior_normalized_width.png',dpi=175);plt.close(fig)
    print('[Audit] Wrote GA recovery, NUTS uncertainty and FIM diagnostic tables and plots 16-17 (NO FIM gates).')



# =============================================================================
# SYNTHETIC DISCOVERY -> HELD-OUT VALIDATION -> UNKNOWN-TRUTH APPLICATION
# =============================================================================
# Legacy combined-gate audit modules were removed in this release.

def load_control_observations(path, sheet_name=None):
    """Return a long-format dataframe from wide XLSX or old long CSV.

    This function only changes input parsing. It neither imputes missing data
    nor calculates a parameter estimate. It strictly checks the six observed
    frequencies and ignores truly blank wide-sheet cells.
    """
    path=Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    suffix=path.suffix.lower()
    if suffix=='.csv':
        dat=pd.read_csv(path,encoding='utf-8-sig')
        needed={'frequency_hz','velocity_um_s'}
        if not needed.issubset(dat.columns):
            raise ValueError('CSV must contain frequency_hz and velocity_um_s columns. '
                             'For the existing six-column Excel layout, supply the .xlsx directly.')
        return dat[['frequency_hz','velocity_um_s']].copy()
    if suffix not in {'.xlsx','.xlsm'}:
        raise ValueError(f'Unsupported control file extension {suffix}. Use .xlsx or a long-format .csv.')
    try:
        import openpyxl
    except ImportError as exc:
        raise RuntimeError('Reading the original Excel file requires openpyxl: pip install openpyxl') from exc
    book=openpyxl.load_workbook(path,read_only=True,data_only=True)
    try:
        if sheet_name is not None:
            if sheet_name not in book.sheetnames:
                raise ValueError(f'Worksheet {sheet_name!r} not found. Available: {book.sheetnames}')
            ws=book[sheet_name]
        else:
            ws=book.worksheets[0]
        ws.reset_dimensions()  # Avoid truncated read-only Excel sheet dimensions
        cells=ws.iter_rows(values_only=True)
        try:
            header=next(cells)
        except StopIteration:
            raise ValueError('Excel sheet is empty.')
        headers=[(j,value) for j,value in enumerate(header)
                 if value is not None and str(value).strip()!='']
        if len(headers)!=len(FREQ_6):
            raise ValueError('Excel first row must contain exactly six frequency headers (Hz). '
                             f'Found {len(headers)}: {[v for _,v in headers]}')
        header_freq=[]
        for j,value in headers:
            if isinstance(value,bool):
                raise ValueError(f'Invalid frequency header at Excel column {j+1}: {value!r}')
            try:
                freq=float(value)
            except (TypeError,ValueError):
                raise ValueError(f'Frequency at Excel column {j+1} must be numeric Hz, got {value!r}')
            if not np.isfinite(freq):
                raise ValueError(f'Frequency at Excel column {j+1} is not finite.')
            header_freq.append(freq)
        match=np.isclose(np.asarray(header_freq)[:,None],FREQ_6[None,:],rtol=1e-6,atol=1e-3)
        if np.any(match.sum(axis=1)!=1) or len(set(match.argmax(axis=1).tolist()))!=6:
            raise ValueError(f'Excel header frequencies must match the six model frequencies in Hz: '
                             f'{FREQ_6.tolist()}. Found: {header_freq}')
        raw=[]
        for excel_row,values in enumerate(cells,start=2):
            for idx,freq in zip([j for j,_ in headers],header_freq):
                value=values[idx] if idx<len(values) else None
                if value is None or (isinstance(value,str) and value.strip()==''):
                    continue
                if isinstance(value,bool):
                    raise ValueError(f'Excel {ws.cell(excel_row,idx+1).coordinate}: boolean is not a velocity.')
                try:
                    velocity=float(value)
                except (ValueError,TypeError):
                    raise ValueError(f'Excel row {excel_row}, column {idx+1}: '
                                     f'velocity must be numeric or blank, got {value!r}')
                if not np.isfinite(velocity):
                    raise ValueError(f'Excel row {excel_row}, column {idx+1}: velocity must be finite.')
                raw.append((freq,velocity))
        if not raw:
            raise ValueError('No numeric DEP velocity observations found below the first Excel row.')
        return pd.DataFrame(raw,columns=['frequency_hz','velocity_um_s'])
    finally:
        book.close()


# Legacy experimental fitting and integrated reporting functions removed.
# Use dep_experiment_v5_integrated.py for experimental inference.


def write_truth_split_file(truths,outdir,a):
    """Lock an outcome-blind split of UNIQUE synthetic truths.

    This replaces the removed legacy audit module which used to create the
    split as a side effect. All replicates/noise levels for a truth stay in
    the SAME split. Existing split files are not overwritten on --resume.
    """
    folder=outdir/'audit'
    folder.mkdir(parents=True,exist_ok=True)
    file=folder/'00_truth_discovery_validation_split.csv'
    ids=set(truths.truth_id.astype(str))
    if file.is_file():
        old=pd.read_csv(file)
        if old.truth_id.astype(str).duplicated().any() or set(old.truth_id.astype(str))!=ids:
            raise ValueError('Existing truth split does not match selected_truths.csv; refusing to overwrite')
        return
    rng=np.random.default_rng(a.audit_split_seed)
    rows=[]
    for label,g in truths.groupby('stratum',sort=True):
        unique=np.array(sorted(g.truth_id.astype(str).unique()),dtype=object)
        rng.shuffle(unique)
        n=len(unique)
        if n==1:
            # With only one truth, no independent held-out validation exists.
            nd=1
        else:
            nd=int(np.clip(round(n*a.audit_train_fraction),1,n-1))
        for i,tid in enumerate(unique):
            rows.append(dict(truth_id=str(tid),stratum=str(label),
                             split='discovery' if i<nd else 'validation'))
    safe_csv_save(pd.DataFrame(rows),file)
    print('[Locked truth split] saved',file)


def main():
    a=parse_args()
    if a.control_raw_xlsx is not None:
        if a.control_raw_csv is not None:
            raise ValueError('Specify only ONE of --control-raw-xlsx and --control-raw-csv.')
        a.control_raw_csv=a.control_raw_xlsx
    outdir=a.output_dir.expanduser().resolve()
    if a.control_raw_csv is not None:
        raise SystemExit('Use dep_experiment_v5_integrated.py for experimental raw data. '
                         'The legacy experiment audit uses retired combined gates.')
    if a.apply_control_csv is not None:
        apply_control_audit(a)
        return
    if a.joint_analysis_only or a.audit_analysis_only or a.apply_control_csv is not None:
        raise SystemExit('Legacy combined-gate audit and joint-gate selection are DISABLED in this release. '
            'Use dep_analysis_v5.py for GA/Bayesian reporting, '
            'or dep_fim_diagnostics_v5.py for independent FIM analysis.')
    cfg={k:(str(v.resolve()) if isinstance(v,Path) else v)
         for k,v in vars(a).items() if k not in (
             "output_dir","resume","dry_run","ga_only","no_auto_postprocess","audit_analysis_only",
             "audit_target_risk","audit_train_fraction","audit_min_accepted","audit_min_accepted_truths",
             "audit_split_seed","audit_bootstrap","recovery_calibration_bins","report_detail","report_quotas","joint_analysis_only","joint_min_accepted_datasets","joint_min_accepted_truths","backfill_joint","backfill_joint_limit","apply_control_csv","control_raw_csv","control_raw_xlsx","control_sheet","audit_rules_json")}
    if a.truths_csv is not None:
        import hashlib
        cfg["truths_csv_sha256"]=hashlib.sha256(a.truths_csv.read_bytes()).hexdigest()
    cfg["freq_grid_hz"]=(FREQ_25 if a.freq_grid=="25" else FREQ_6).tolist()
    cfg["model"]="single-shell 4 dielectric parameters + sigma_y HalfCauchy(5) for NUTS"
    cfg["fim_parameter_coordinates"]="ln dielectric parameters"
    cfg["fim_formula"]="J.T diag(n_per_frequency / sigma_reference^2) J; 4x4"
    cfg["fim_information"]="Primary: log10(lambda_min(F)) weakest direction; secondary: log10 det(F)/4 geometric eigenvalue mean. F in ln(theta) coordinates."
    cfg["info_class_definition"]="Terciles among selected truths at sigma-reference; actual-noise levels classified with unchanged numeric cuts."
    cfg["3d_plot_axes"]="x=log10 kappa(F), y=log10 lambda_min(F at generating SD), z=mean absolute log10 recovery error"
    cfg["prior_interpretation"]="Independent broad priors, NOT biological population frequencies"
    cfg["three_evaluations"]="Recovery accuracy; posterior 95% truth coverage (synthetic only, pass NUTS); uncertainty precision: CI95 width / log10-prior-span (%)"
    cfg["noise_adjusted_FIM"]="Per-parameter sqrt(diag(F^-1)) in ln theta from full SVD at synthetic oracle and practical GA estimate; GA sigma estimated from residuals"
    cfg["noise_and_inference"]="Noise unknown to NUTS; GA sigma MLE from residuals"
    cfg["comparison"]="GA minimizes full-observation Gaussian SSE (mean SSE differs by constant); NUTS uses same observations"
    cfg["notes"]=("Prior mode includes all FIM regimes (including S4); "
                  "stratified mode excludes S4. Conditional CI inclusion need not equal 95%; "
                  "GA multistart variability is NOT uncertainty")
    cfg["controlled_design"]=("Unconditional independent log-uniform sampling with post-hoc FIM labels, "
                              "including S4" if a.sampling_mode == 'prior' else
                              "S1 < first cutoff, S2 [first,second), S3 [second,third); S4 >=third excluded")
    cfg["noise_default"]="fixed sigma_y=5 um/s; noise multipliers=HalfCauchy(5) quartile ratios; NUTS sigma is INFERRED, not fixed"
    cfg["noise_scenario_interpretation"]="Prior-quantile sensitivity scenarios, NOT measured or validated experimental noise levels"
    cfg["spectrum_curve_warning"]="Dense-frequency curves are interpolated forward-model evaluations; data were measured/simulated ONLY at FREQ_6"
    cfgfile=outdir/"config.json"
    if outdir.exists() and any(outdir.iterdir()):
        if not a.resume:
            raise SystemExit(f"Output directory {outdir} is not empty; use another path or --resume.")
        if not cfgfile.exists() or json.loads(cfgfile.read_text(encoding="utf8"))!=cfg:
            raise SystemExit("Cannot resume: config.json differs; use exact same settings.")
    else:
        outdir.mkdir(parents=True,exist_ok=True)
        cfgfile.write_text(json.dumps(cfg,indent=2,ensure_ascii=False),encoding="utf8")
    frequencies=FREQ_25 if a.freq_grid=="25" else FREQ_6
    pred,jac_batch=make_forward_and_jac(frequencies)
    if a.resume:
        pool=pd.read_csv(outdir/"screened_prior.csv")
        truths=pd.read_csv(outdir/"selected_truths.csv")
        edges,labels=stratum_labels(a.fim_bounds, include_s4=(a.sampling_mode == "prior"))
    else:
        pool,truths,edges,labels=screen_and_select(a,frequencies,outdir,pred,jac_batch)
    # Selection is based on synthetic truths only; split is independent of GA/NUTS outcomes.
    write_truth_split_file(truths,outdir,a)
    # Mandatory fixed bin boundaries: never dynamically re-bin by outcome.
    bins_meta=json.loads((outdir/"information_reference_bins.json").read_text(encoding="utf8"))
    info_cuts=(float(bins_meta["log10_lambda_min_lower_cutoff"]),
               float(bins_meta["log10_lambda_min_upper_cutoff"]))
    geom_cuts=(float(bins_meta["log10_geom_mean_lower_cutoff"]),
               float(bins_meta["log10_geom_mean_upper_cutoff"]))
    anchor_jac=np.asarray(jac_batch(jnp.asarray(np.log(EXP_ANCHOR)[None,:])))[0]
    anchor_info=svd_information(anchor_jac,a.n_per_frequency,a.sigma_y)
    print(f"[Anchor reference, synthetic grid] FIM cond={anchor_info['fim_cond']:.6g}, "
          f"Jacobian cond={anchor_info['cond_j']:.6g}; NOT generating truth.")
    (outdir/"anchor_fim_reference.json").write_text(json.dumps(
        {k:float(anchor_info[k]) for k in ("fim_cond","log10_fim_cond","cond_j","fim_lmin","fim_lmax","fim_logdet")},indent=2))
    if a.dry_run:
        print("[Noise] generation mode=",a.sigma_generation," base sigma=",a.sigma_y," multipliers=",a.noise_multipliers)
        plot_results(pool,truths,pd.DataFrame(),pd.DataFrame(),labels,edges,outdir)
        print(f"[Done: screening only] {len(truths)} selected truths; output: {outdir}")
        return
    if not a.ga_only:
        try:
            import numpyro  # noqa:F401
            import arviz  # noqa:F401
        except ImportError as exc:
            raise SystemExit("Install numpyro/arviz for paired NUTS, or run --ga-only.") from exc
    # GA and NUTS checkpoint independently: --ga-only can later be resumed with NUTS.
    ga_dpath=outdir/"results_GA_datasets.csv"
    ga_ppath=outdir/"results_GA_parameters.csv"
    ga_rpath=outdir/"results_GA_runs.csv"
    nuts_dpath=outdir/"results_datasets.csv"
    nuts_ppath=outdir/"results_parameters.csv"
    ga_drows=_load_csv_records(ga_dpath)
    ga_prows=_load_csv_records(ga_ppath)
    ga_rrows=_load_csv_records(ga_rpath)
    nuts_drows=_load_csv_records(nuts_dpath)
    nuts_prows=_load_csv_records(nuts_ppath)
    ga_completed={r["dataset_id"] for r in ga_drows}
    nuts_completed={r["dataset_id"] for r in nuts_drows}
    if a.backfill_joint:
        import shutil
        for src in [nuts_dpath,nuts_ppath]:
            if src.is_file():
                backup=src.with_name(src.stem+'.pre_joint_backfill.csv')
                if not backup.exists():
                    shutil.copy2(src,backup)
                    print('[JOINT backfill] Original NUTS data backed up:',backup)
    existing_nuts={str(r['dataset_id']):r for r in nuts_drows}
    backfill_started=0
    jobs=[]
    for i,(_,truth) in enumerate(truths.iterrows()):
        theta=truth[list(PARAM_NAMES)].to_numpy(dtype=float)
        truth_id=str(truth.truth_id)
        sigma_rng=np.random.default_rng(np.random.SeedSequence([a.seed,333,i]))
        sigma_base=(float(5.0*abs(sigma_rng.standard_cauchy())) if a.sigma_generation=="prior" else a.sigma_y)
        if not np.isfinite(sigma_base) or sigma_base<=0:
            raise RuntimeError(f"Invalid synthetic true sigma for {truth_id}: {sigma_base}")
        velocity_true=np.asarray(pred(jnp.asarray(np.log(theta))),dtype=float)
        for j in range(a.datasets_per_truth):
            yrng=np.random.default_rng(np.random.SeedSequence([a.seed,444,i,j]))
            z=yrng.standard_normal(size=(a.n_per_frequency,len(frequencies)))
            for l,mult in enumerate(a.noise_multipliers):
                sigma_true=sigma_base*mult
                dataset_id=f"{truth_id}_D{j+1:03d}_L{l+1:02d}"
                mcmc_seed=int(np.random.default_rng(np.random.SeedSequence([a.seed,555,i,j,l])).integers(0,2**31-1))
                ga_seed=int(np.random.default_rng(np.random.SeedSequence([a.seed,666,i,j,l])).integers(0,2**31-1))
                jobs.append((dataset_id,truth_id,str(truth.stratum),theta,truth,sigma_true,velocity_true,z,mcmc_seed,ga_seed,mult))
    # Fixed seed streams: exactly the same Normal z is used for low/medium/high
    # noise at a given truth/replicate.  Record observed means and SDs for plots.
    spectra_rows=[]
    for job in jobs:
        did,tid,lab,theta,trow,sig_true,v_true,z,_,_,mult=job
        yy=v_true[None,:]+sig_true*z
        for ki,fk in enumerate(frequencies):
            spectra_rows.append(dict(dataset_id=did,truth_id=tid,stratum=lab,
                sigma_true=float(sig_true),noise_multiplier=float(mult),
                frequency_hz=float(fk),true_velocity=float(v_true[ki]),
                observed_mean=float(np.mean(yy[:,ki])),
                observed_sd=float(np.std(yy[:,ki],ddof=1)) if len(yy)>1 else float("nan"),
                n_observations=int(len(yy))))
    safe_csv_save(pd.DataFrame(spectra_rows),outdir/"spectra_observed_summary.csv")
    # Oracle FIM local-Gaussian width estimates for each truth/noise scenario.
    # These are NOT posterior credible intervals; useful for separating
    # FIM information scale from condition number and for optional post hoc checks.
    fim_width_rows=[]
    fim_info_rows=[]
    for _,truth in truths.iterrows():
        th=truth[list(PARAM_NAMES)].to_numpy(dtype=float)
        JJ=np.asarray(jac_batch(jnp.asarray(np.log(th)[None,:])))[0]
        related=[q for q in jobs if q[1]==str(truth.truth_id)]
        for noise_mult in a.noise_multipliers:
            options=[q for q in related if np.isclose(q[-1],noise_mult)]
            if not options: continue
            sd=float(options[0][5])
            info=svd_information(JJ,a.n_per_frequency,sd)
            _,sing,vt=np.linalg.svd(info["Jw"],full_matrices=False)
            if np.min(sing)>0 and np.all(np.isfinite(sing)):
                log_sd=np.sqrt(np.sum((vt.T/sing)**2,axis=1))
                widths=2*1.95996398454*log_sd/np.log(10)
            else:
                widths=np.full(4,np.nan)
            info_group=label_information(info["log10_lambda_min"],info_cuts)
            geom_group=label_information(info["log10_fim_info_geom"],geom_cuts)
            fim_info_rows.append(dict(
                truth_id=str(truth.truth_id),stratum=str(truth.stratum),
                info_class_reference=str(truth.info_class_reference),
                info_class_at_true_sigma=info_group,
                info_geom_class_reference=str(truth.info_geom_class_reference),
                info_geom_class_at_true_sigma=geom_group,
                fim_sigma_reference=float(a.sigma_y),sigma_true=sd,
                noise_multiplier=float(noise_mult),
                log10_fim_cond=float(info["log10_fim_cond"]),
                fim_condition_number=float(info["fim_cond"]),
                log10_lambda_min_reference=float(truth.log10_lambda_min),
                log10_lambda_min_true_sigma=float(info["log10_lambda_min"]),
                log10_lambda_max_true_sigma=float(info["log10_lambda_max"]),
                log10_fim_info_geom_reference=float(truth.log10_fim_info_geom),
                log10_fim_info_geom_true_sigma=float(info["log10_fim_info_geom"]),
                log10_fim_trace_true_sigma=float(info["log10_fim_trace"]),
                fim_logdet_true_sigma=float(info["fim_logdet"]),
                log10_information_ref_q33=info_cuts[0],
                log10_information_ref_q67=info_cuts[1]))
            for i,name in enumerate(PARAM_NAMES):
                pw=float(np.log10(HIGH[i]/LOW[i]))
                fim_width_rows.append(dict(truth_id=str(truth.truth_id),stratum=str(truth.stratum),
                    parameter=name,noise_multiplier=float(noise_mult),sigma_true=sd,
                    log10_fim_cond=float(info["log10_fim_cond"]),
                    log10_lambda_min=float(info["log10_lambda_min"]),
                    log10_fim_info_geom_true_sigma=float(info["log10_fim_info_geom"]),
                    info_class_reference=str(truth.info_class_reference),
                    info_class_at_true_sigma=info_group,
                    info_geom_class_reference=str(truth.info_geom_class_reference),
                    info_geom_class_at_true_sigma=geom_group,
                    fim_predicted_95_width_log10=float(widths[i]),
                    prior_log10_span=pw,
                    fim_width_over_prior=float(widths[i]/pw),
                    oracle_fim_sd_ln=float(log_sd[i]),
                    oracle_fim_sd_log10=float(log_sd[i]/math.log(10)),
                    oracle_fim_95_halfwidth_ln=float(Z_95*log_sd[i]),
                    oracle_fim_95_width_percent_prior=float(100*widths[i]/pw),
                    oracle_fim_precision_Q_factor2=float(Z_95*log_sd[i]/math.log(2))))
    safe_csv_save(pd.DataFrame(fim_width_rows),outdir/"FIM_local_widths_by_truth_and_noise.csv")
    safe_csv_save(pd.DataFrame(fim_info_rows),outdir/"FIM_condition_information_by_truth_noise.csv")
    print(f"[Noise] generation mode={a.sigma_generation}, base sigma_y={a.sigma_y}, multipliers={a.noise_multipliers}")
    print(f"[NUTS configuration] dense_mass={a.dense_mass}, target_accept={a.target_accept}, "
          f"max_tree_depth={a.max_tree_depth}, chains={a.chains}, warmup={a.warmup}, samples={a.samples}")
    print(f"[Plan] {len(truths)} independent truths; {len(jobs)} paired datasets; "
          f"GA multistart={a.ga_runs}, population={a.ga_population}, generations={a.ga_generations}; "
          f"NUTS {'DISABLED (GA-only)' if a.ga_only else str(a.chains)+' chains'}")
    for counter,job in enumerate(jobs,1):
        dataset_id,truth_id,lab,theta,truth,sig_true,v_true,z,mcmc_seed,ga_seed,mult=job
        need_ga=dataset_id not in ga_completed
        old=existing_nuts.get(dataset_id,{})
        target_recorded=pd.to_numeric(old.get('posterior_joint_target_factor_used',np.nan),errors='coerce')
        has_exact_joint=(np.isfinite(pd.to_numeric(old.get('posterior_joint_prob_recovery_GA_center',np.nan),errors='coerce'))
                         and np.isfinite(pd.to_numeric(old.get('posterior_joint_prob_recovery_Bayes_center',np.nan),errors='coerce'))
                         and np.isfinite(target_recorded)
                         and abs(float(target_recorded)-a.audit_target_factor)<1e-10)
        backfill_candidate=(a.backfill_joint and dataset_id in nuts_completed and not has_exact_joint)
        if backfill_candidate and a.backfill_joint_limit and backfill_started>=a.backfill_joint_limit:
            backfill_candidate=False
        need_nuts=(not a.ga_only) and ((dataset_id not in nuts_completed) or backfill_candidate)
        if backfill_candidate:backfill_started+=1
        if not (need_ga or need_nuts):
            print(f"[Skip {counter}/{len(jobs)}] {dataset_id} already completed")
            continue
        y=v_true[None,:]+sig_true*z
        common=dict(dataset_id=dataset_id,truth_id=truth_id,stratum=lab,
                    log10_fim_cond=float(truth.log10_fim_cond),
                    log10_lambda_min=float(truth.log10_lambda_min),
                    log10_lambda_min_true_sigma=float(truth.log10_lambda_min) +
                        2.0*math.log10(a.sigma_y / sig_true),
                    log10_fim_info_geom_reference=float(truth.log10_fim_info_geom),
                    log10_fim_info_geom_true_sigma=float(truth.log10_fim_info_geom)+
                        2.0*math.log10(a.sigma_y/sig_true),
                    log10_lambda_max_true_sigma=float(truth.log10_lambda_max)+
                        2.0*math.log10(a.sigma_y/sig_true),
                    log10_fim_trace_reference=float(truth.log10_fim_trace),
                    log10_fim_trace_true_sigma=float(truth.log10_fim_trace)+
                        2.0*math.log10(a.sigma_y/sig_true),
                    fim_condition_number=float(10.**float(truth.log10_fim_cond)),
                    info_class_reference=str(truth.info_class_reference),
                    info_class_at_true_sigma=label_information(
                        float(truth.log10_lambda_min)+2.0*math.log10(a.sigma_y/sig_true),info_cuts),
                    info_geom_class_reference=str(truth.info_geom_class_reference),
                    info_geom_class_at_true_sigma=label_information(
                        float(truth.log10_fim_info_geom)+2.0*math.log10(a.sigma_y/sig_true),geom_cuts),
                    log10_information_ref_q33=info_cuts[0],
                    log10_information_ref_q67=info_cuts[1],
                    fim_sigma_reference=float(a.sigma_y),
                    sigma_generation=a.sigma_generation,noise_multiplier=float(mult),
                    sigma_true=float(sig_true),n_per_frequency=a.n_per_frequency)
        print(f"[Run {counter}/{len(jobs)}] {dataset_id}, "
              f"FIM log10cond={truth.log10_fim_cond:.3f}, sigma_true={sig_true:.4g}",flush=True)
        if need_ga:
            try:
                ga_d,ga_p,ga_runs=run_ga_multistart(a,y,theta,sig_true,v_true,frequencies,ga_seed)
                ga_drows.append({**common,**ga_d})
                ga_prows.extend([{**common,**r} for r in ga_p])
                ga_rrows.extend([{**common,**r} for r in ga_runs])
                print(f"  GA best run={ga_d['ga_best_run']}, "
                      f"abs log10 error={ga_d['ga_mean_abs_param_log10_error']:.3f}, "
                      f"velocity RMSE truth={ga_d['ga_velocity_rmse_to_true']:.3f}",flush=True)
            except Exception as exc:
                import traceback
                ga_drows.append({**common,"ga_status":"error","ga_exception":str(exc)[:500],
                                 "ga_traceback":traceback.format_exc()[-1500:]})
                print(f"  [GA ERROR] {type(exc).__name__}: {exc}",flush=True)
            safe_csv_save(pd.DataFrame(ga_drows),ga_dpath)
            if ga_prows:safe_csv_save(pd.DataFrame(ga_prows),ga_ppath)
            if ga_rrows:safe_csv_save(pd.DataFrame(ga_rrows),ga_rpath)
        if need_nuts:
            try:
                ga_match=next((r for r in reversed(ga_drows)
                               if r.get('dataset_id')==dataset_id and r.get('ga_status')=='ok'),None)
                ga_hat=(np.array([ga_match['ga_best_'+name] for name in PARAM_NAMES],float)
                        if ga_match is not None else None)
                data,p_rows=run_one_mcmc(a,y,theta,sig_true,v_true,frequencies,mcmc_seed,outdir,dataset_id,
                                         ga_theta=ga_hat)
                # Replace original dataset atomically in the in-memory CSV
                # lists only AFTER NUTS succeeded; no duplicate IDs.
                nuts_drows[:]=[r for r in nuts_drows if r.get('dataset_id')!=dataset_id]
                nuts_prows[:]=[r for r in nuts_prows if r.get('dataset_id')!=dataset_id]
                nuts_drows.append({**common,**data})
                nuts_prows.extend([{**common,**r,"mcmc_pass":data["mcmc_pass"]} for r in p_rows])
                print(f"  NUTS pass={data['mcmc_pass']}, Rhat max={data['max_rhat']:.4f}, "
                      f"divergences={data['n_divergences']}, "
                      f"abs log10 error={data['mean_abs_param_log10_error']:.3f}",flush=True)
            except Exception as exc:
                import traceback
                if dataset_id not in existing_nuts:
                    nuts_drows.append({**common,"status":"error","mcmc_pass":False,
                                       "exception":str(exc)[:500],"traceback":traceback.format_exc()[-1500:]})
                else:
                    print('  [JOINT backfill] Preserved old NUTS row after refit failure')
                print(f"  [NUTS ERROR] {type(exc).__name__}: {exc}",flush=True)
            safe_csv_save(pd.DataFrame(nuts_drows),nuts_dpath)
            if nuts_prows:safe_csv_save(pd.DataFrame(nuts_prows),nuts_ppath)
    ga_d=pd.DataFrame(ga_drows);ga_p=pd.DataFrame(ga_prows)
    nuts_d=pd.DataFrame(nuts_drows);nuts_p=pd.DataFrame(nuts_prows)
    stats,matched_d,matched_p=_summarize_comparison(ga_d,ga_p,nuts_d,nuts_p,truths,labels,outdir)
    # Keep original NUTS-only summary and plots for comparison with prior script.
    if not nuts_d.empty:
        safe_csv_save(make_summary(nuts_d,nuts_p,truths,labels,a),outdir/"summary_by_stratum.csv")
    plot_results(pool,truths,nuts_d,nuts_p,labels,edges,outdir)
    ggood=ga_d[ga_d.ga_status=="ok"] if not ga_d.empty else pd.DataFrame()
    gp=ga_p[(ga_p.parameter!="sigma_y") & (ga_p.dataset_id.isin(ggood.dataset_id))] if not ga_p.empty else pd.DataFrame()
    _plot_ga_nuts_comparison(matched_d,matched_p,ggood,gp,labels,outdir)
    make_noise_stratum_summary(ga_d,nuts_d,nuts_p,labels,outdir)
    make_information_recovery_outputs(ga_d,nuts_d,labels,outdir)
    build_three_evaluation_outputs(a,ga_d,ga_p,nuts_d,nuts_p,
                                   pd.DataFrame(fim_width_rows),jac_batch,outdir)
    # Deliberately removed the former automatic reliability_audit(): it created
    # GA+FIM, GA+FIM+Bayesian combined reporting and post-hoc threshold rules.
    # FIM remains available as an independent identifiability DIAGNOSTIC.
    plot_fit_vs_recovery(ga_d,labels,outdir)
    make_spectra_plots(a,ga_d,ga_p,nuts_d,nuts_p,truths,outdir)
    print("\n=== GA vs NUTS BY FIM STRATUM ===")
    print(stats.to_string(index=False))
    print("\n[Method] GA finds a multistart SSE-based point estimate; NUTS infers the full posterior.")
    print("[Fairness] For every dataset, both methods see exactly the same observations and model.")
    print("[Caution] GA multistart spread measures optimization variability; it is NOT a GA CI.")
    print("[Caution] Paired inference comparisons use only NUTS runs passing all diagnostics.")
    print("[Caution] Oracle FIM widths use generating theta and sigma; they are LOCAL Gaussian approximations, not posterior intervals.")
    print("[Caution] Conditioning strata are not biological prevalences; conditional CI inclusion is descriptive.")
    print(f"[Done] {outdir}")


if __name__=="__main__":
    main()
    # Run BOTH independent postprocess jobs after a completed non-dry-run run.
    # Each writes to a documented subfolder inside the requested --output-dir.
    # A status file records SUCCESS/FAILED independently; silent failures are prohibited.
    _args = parse_args()
    _directory = Path(_args.output_dir).expanduser().resolve()
    if _args.dry_run:
        print('[POSTPROCESS SKIPPED] --dry-run creates only screening outputs.')
    elif _args.no_auto_postprocess:
        print('[POSTPROCESS SKIPPED] --no-auto-postprocess requested.')
    elif not (_directory / 'results_GA_datasets.csv').exists():
        print('[POSTPROCESS SKIPPED] results_GA_datasets.csv does not exist.')
    else:
        from dep_postprocess_v5 import run_postprocess
        _status = run_postprocess([_directory], _directory,
                                  seed_labels=[f'seed{_args.seed}'],
                                  split='all', bootstrap=200,
                                  recompute_correlations=True)
        print('[POSTPROCESS STATUS] ', _directory / 'POSTPROCESS_STATUS.json')
        if any(v['state'] != 'SUCCESS' for v in _status['tasks'].values()):
            print('[POSTPROCESS INCOMPLETE] GA/NUTS outputs were preserved. '
                  'Inspect POSTPROCESS_STATUS.json and each subfolder ERROR.txt.')
