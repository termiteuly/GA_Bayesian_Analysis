# -*- coding: utf-8 -*-
# ==========================================================
# DEP Bayesian pipeline + Point-wise Train/Test split
# Added: TRAIN metrics (ELPD/RMSE/Coverage) + ELPD gap
# ASCII-only output.
# ==========================================================

import os
import time
os.environ["JAX_PLATFORM_NAME"] = os.getenv("JAX_PLATFORM_NAME", "cpu")

import numpy as onp
SEED = int(os.getenv("SEED", "0"))
onp.random.seed(SEED)

# ----- MCMC config -----
NCHAINS        = int(os.getenv("NCHAINS", "5"))
CHAIN_METHOD   = os.getenv("CHAIN_METHOD", "sequential")  # sequential | vectorized | parallel
TARGET_ACCEPT  = float(os.getenv("TARGET_ACCEPT", "0.99"))
NUM_WARMUP     = int(os.getenv("NUM_WARMUP", "15000"))
NUM_SAMPLES    = int(os.getenv("NUM_SAMPLES", "35000"))
MAX_TREE_DEPTH = int(os.getenv("MAX_TREE_DEPTH", "10"))

# ----- Split / evaluation config -----
TEST_FRACTION   = float(os.getenv("TEST_FRACTION", "0.20"))  # hold out fraction PER FREQUENCY
N_SPLITS        = int(os.getenv("N_SPLITS", "5"))
MAX_POST_SAMPLES_FOR_EVAL = int(os.getenv("EVAL_POST_SAMPLES", "4000"))
M_NOISE_DRAWS   = int(os.getenv("M_NOISE_DRAWS", "20"))

# ----- Paths -----
RAW_STATS_CSV = os.getenv("RAW_STATS_CSV", "raw_per_frequency_mean_sd.csv")
DATA_XLSX     = os.getenv("DATA_XLSX", "/home/cjli/Python/MCMCpy/DEPvelocityRAW_Control.xlsx")

# ----- Packages -----
import pandas as pd

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as random

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.infer.initialization import init_to_feasible
import numpyro.handlers as handlers
import arviz as az


# ==========================================================
# Constants
# ==========================================================
eps0   = 8.854e-12
E      = 30000.0
D      = 1e-3
r      = 3.0e-6 + 55e-9
tmem   = 5.0e-9
tw     = 50.0e-9
t_shell = tmem + tw
ep_med = 80 * eps0
si_med = 0.1

# ----- Priors (log-space Uniform) -----
log_eps_cyto_low    = jnp.log(10*eps0);   log_eps_cyto_high    = jnp.log(200*eps0)
log_sigma_cyto_low  = jnp.log(1e-6);      log_sigma_cyto_high  = jnp.log(1.0)
log_eps_shell_low   = jnp.log(10*eps0);   log_eps_shell_high   = jnp.log(200*eps0)
log_sigma_shell_low = jnp.log(1e-6);      log_sigma_shell_high = jnp.log(1.0)


# ==========================================================
# Data loading
# ==========================================================
def load_one_xlsx(path):
    df = pd.read_excel(path, header=None)
    freq_raw = onp.asarray(df.iloc[0].values, dtype=float)
    V_raw    = onp.asarray(df.iloc[1:].values, dtype=float)

    col_ok = onp.isfinite(freq_raw) & onp.any(onp.isfinite(V_raw), axis=0)
    freq = freq_raw[col_ok]
    V    = V_raw[:, col_ok]

    ord_idx = onp.argsort(freq)
    freq = freq[ord_idx]
    V    = V[:, ord_idx]

    mask_obs_np = onp.isfinite(V)
    V_filled_np = onp.where(mask_obs_np, V, 0.0)
    return freq, V, mask_obs_np, V_filled_np

freq, V, mask_obs_np, V_filled_np = load_one_xlsx(DATA_XLSX)

Exp_Frequency = jnp.array(freq)              # (K,)
V_filled      = jnp.array(V_filled_np)       # (Ncells,K)
mask_obs      = jnp.array(mask_obs_np)       # (Ncells,K)

Ncells, K = V.shape
print("[Data] cells N=%d, frequencies K=%d" % (Ncells, K))


# ==========================================================
# Raw per-frequency stats + within-frequency pooled SD (ALL points)
# ==========================================================
def raw_stats_per_frequency_and_within(freq_np, V_np, mask_np):
    freq_np = onp.asarray(freq_np, dtype=float)
    V_np = onp.asarray(V_np, dtype=float)
    mask_np = onp.asarray(mask_np).astype(bool)

    V_nan = onp.where(mask_np & onp.isfinite(V_np), V_np, onp.nan)

    n_k = onp.sum(onp.isfinite(V_nan), axis=0).astype(int)
    mean_k = onp.nanmean(V_nan, axis=0)
    sd_k = onp.nanstd(V_nan, axis=0, ddof=1)
    sem_k = onp.where(n_k > 0, sd_k / onp.sqrt(onp.maximum(n_k, 1)), onp.nan)

    df = pd.DataFrame({
        "freq_Hz": freq_np,
        "n": n_k,
        "mean": mean_k,
        "sd": sd_k,
        "sem": sem_k,
    })

    denom = int(onp.sum(onp.maximum(n_k - 1, 0)))
    if denom <= 0:
        within_sd = float("nan")
    else:
        resid = V_nan - mean_k[None, :]
        sse = onp.nansum(resid**2)
        within_sd = float(onp.sqrt(sse / denom))

    return df, within_sd

raw_df, within_sd_all = raw_stats_per_frequency_and_within(freq, V, mask_obs_np)

print("\n[Raw data per-frequency summary] (ALL points)")
print("%3s  %12s  %4s  %12s  %12s  %12s" % ("k", "freq(Hz)", "n", "mean", "sd", "sem"))
for k in range(K):
    row = raw_df.iloc[k]
    print("%3d  %12.4e  %4d  %12.4e  %12.4e  %12.4e" % (
        k, row["freq_Hz"], int(row["n"]), row["mean"], row["sd"], row["sem"]
    ))

raw_df.to_csv(RAW_STATS_CSV, index=False)
print("\nSaved raw per-frequency stats -> %s" % RAW_STATS_CSV)

print("\n[Raw data within-frequency pooled SD] (ALL points)")
print("within_sd_all = %.4e" % within_sd_all)


# ==========================================================
# Physics model
# ==========================================================
def eps_complex(f, eps_r, sigma):
    return eps_r - 1j * sigma / (2 * jnp.pi * f + 1e-300)

def fun_cytoshell(f, eps_cyto, sigma_cyto, eps_shell, sigma_shell):
    eps_i = eps_complex(f, eps_cyto,  sigma_cyto)
    eps_o = eps_complex(f, eps_shell, sigma_shell)
    a = (r / (r - t_shell))
    num = (a**3 + 2.0 * (eps_i - eps_o) / (eps_i + 2.0 * eps_o + 1e-300))
    den = (a**3 -       (eps_i - eps_o) / (eps_i + 2.0 * eps_o) )
    den = den + 1e-30j
    return eps_o * (num / (den + 1e-300))

def fun_fcm(f, eps_cyto, sigma_cyto, eps_shell, sigma_shell):
    ep_med_c = eps_complex(f, ep_med, si_med)
    ep_p     = fun_cytoshell(f, eps_cyto, sigma_cyto, eps_shell, sigma_shell)
    return (ep_p - ep_med_c) / (ep_p + 2.0 * ep_med_c + 1e-300)

def theory_velocity(f, eps_cyto, sigma_cyto, eps_shell, sigma_shell):
    dep_const = ((r**2 * ep_med * (E**2 / 9e-6)) / (3.0 * D)) * 1e6
    return dep_const * jnp.real(fun_fcm(f, eps_cyto, sigma_cyto, eps_shell, sigma_shell))


# ==========================================================
# NumPyro model
# ==========================================================
def model_freq_only_notau(f_obs, V_obs, mask_obs):
    log_eps_cyto    = numpyro.sample("log_eps_cyto",    dist.Uniform(log_eps_cyto_low,    log_eps_cyto_high))
    log_sigma_cyto  = numpyro.sample("log_sigma_cyto",  dist.Uniform(log_sigma_cyto_low,  log_sigma_cyto_high))
    log_eps_shell   = numpyro.sample("log_eps_shell",   dist.Uniform(log_eps_shell_low,   log_eps_shell_high))
    log_sigma_shell = numpyro.sample("log_sigma_shell", dist.Uniform(log_sigma_shell_low, log_sigma_shell_high))

    eps_cyto    = jnp.exp(log_eps_cyto)
    sigma_cyto  = jnp.exp(log_sigma_cyto)
    eps_shell   = jnp.exp(log_eps_shell)
    sigma_shell = jnp.exp(log_sigma_shell)

    v_pred = theory_velocity(f_obs, eps_cyto, sigma_cyto, eps_shell, sigma_shell)  # (K,)
    sigma_y = numpyro.sample("sigma_y", dist.HalfCauchy(5.0))

    with handlers.mask(mask=mask_obs):
        numpyro.sample("obs", dist.Normal(v_pred[None, :], sigma_y), obs=V_obs)


# ==========================================================
# Split: point-wise stratified by frequency column
# ==========================================================
def stratified_point_split(mask_obs_np_in, test_fraction, seed):
    rng = onp.random.default_rng(seed)
    Ncells_local, K_local = mask_obs_np_in.shape
    mask_obs_np_in = onp.asarray(mask_obs_np_in).astype(bool)

    mask_test = onp.zeros((Ncells_local, K_local), dtype=bool)

    for k in range(K_local):
        rows_k = onp.where(mask_obs_np_in[:, k])[0]
        n_k = int(rows_k.size)
        if n_k <= 1:
            continue

        n_test = int(onp.floor(n_k * test_fraction))
        n_test = max(1, n_test) if n_k >= 2 else 0
        if n_test >= n_k:
            n_test = n_k - 1

        sel = rng.choice(rows_k, size=n_test, replace=False)
        mask_test[sel, k] = True

    mask_train = mask_obs_np_in & (~mask_test)

    # TRAIN per-frequency SD_k (for NRMSE_freq; no test leakage)
    sd_k_train = onp.full((K_local,), onp.nan, dtype=float)
    for k in range(K_local):
        vals = V[:, k]
        m = mask_train[:, k] & onp.isfinite(vals)
        x = vals[m]
        if x.size >= 2:
            sd_k_train[k] = float(onp.std(x, ddof=1))

    return mask_train, mask_test, sd_k_train


# ==========================================================
# Posterior subsampling
# ==========================================================
def subsample_posterior(posterior_samples, max_keep, seed):
    S = int(next(iter(posterior_samples.values())).shape[0])
    keep = min(S, int(max_keep))
    rng = onp.random.default_rng(seed)
    sel = rng.choice(S, size=keep, replace=False)
    return {k: v[sel] for k, v in posterior_samples.items()}


# ==========================================================
# Metrics on an arbitrary mask (TRAIN or TEST)
# ==========================================================
def compute_elpd_rmse_cov(posterior_samples, f_obs, V_np, mask_eval_np, seed, sd_k_train):
    post = subsample_posterior(posterior_samples, MAX_POST_SAMPLES_FOR_EVAL, seed)
    post_np = {k: onp.asarray(v) for k, v in post.items()}

    eps_cyto   = onp.exp(post_np["log_eps_cyto"])
    sig_cyto   = onp.exp(post_np["log_sigma_cyto"])
    eps_shell  = onp.exp(post_np["log_eps_shell"])
    sig_shell  = onp.exp(post_np["log_sigma_shell"])
    sigma_y    = post_np["sigma_y"]
    S = int(sigma_y.shape[0])

    # latent curve per posterior draw (S,K)
    v_lat_list = []
    for i in range(S):
        v_i = theory_velocity(
            f_obs,
            jnp.array(eps_cyto[i]),
            jnp.array(sig_cyto[i]),
            jnp.array(eps_shell[i]),
            jnp.array(sig_shell[i]),
        )
        v_lat_list.append(onp.asarray(v_i))
    v_lat_all = onp.stack(v_lat_list, axis=0)  # (S,K)

    V_np = onp.asarray(V_np, dtype=float)
    mask_eval_np = onp.asarray(mask_eval_np).astype(bool)

    obs_rows, obs_cols = onp.where(mask_eval_np & onp.isfinite(V_np))
    y_obs = V_np[obs_rows, obs_cols]
    Nobs = int(y_obs.size)

    if Nobs == 0:
        return {
            "n_points": 0,
            "elpd_per_point": onp.nan,
            "rmse": onp.nan,
            "cov50": onp.nan,
            "cov95": onp.nan,
            "rmse_per_freq": onp.full((K,), onp.nan),
            "n_per_freq": onp.zeros((K,), dtype=int),
            "nrmse_freq": onp.nan,
        }

    # ELPD/point
    mu_s  = v_lat_all[:, obs_cols]     # (S,Nobs)
    sig_s = sigma_y[:, None]          # (S,1)
    y     = y_obs[None, :]            # (1,Nobs)

    logN = -0.5*onp.log(2.0*onp.pi) - onp.log(sig_s) - 0.5*((y - mu_s)/sig_s)**2
    m = onp.max(logN, axis=0)
    lpd = m + onp.log(onp.mean(onp.exp(logN - m[None, :]), axis=0))
    elpd_per_point = float(onp.mean(lpd))

    # RMSE (posterior mean prediction)
    yhat = onp.mean(mu_s, axis=0)
    rmse = float(onp.sqrt(onp.mean((y_obs - yhat)**2)))

    # RMSE per frequency
    rmse_k = onp.full((K,), onp.nan, dtype=float)
    n_k = onp.zeros((K,), dtype=int)
    for kk in range(K):
        mkk = (obs_cols == kk)
        nkk = int(onp.sum(mkk))
        n_k[kk] = nkk
        if nkk > 0:
            err = y_obs[mkk] - yhat[mkk]
            rmse_k[kk] = float(onp.sqrt(onp.mean(err**2)))

    # NRMSE_freq standardized by TRAIN SD_k (no leakage)
    sd_k_train = onp.asarray(sd_k_train, dtype=float)
    sd_at_pts = sd_k_train[obs_cols]
    good = onp.isfinite(sd_at_pts) & (sd_at_pts > 0)
    if onp.any(good):
        z = (y_obs[good] - yhat[good]) / sd_at_pts[good]
        nrmse_freq = float(onp.sqrt(onp.mean(z**2)))
    else:
        nrmse_freq = float("nan")

    # Coverage via posterior predictive mixture sampling
    rng = onp.random.default_rng(seed + 12345)
    M = max(1, int(M_NOISE_DRAWS))

    idx_post = rng.integers(0, S, size=S * M)
    mu_points = v_lat_all[:, obs_cols]       # (S,Nobs)
    mu_draw   = mu_points[idx_post, :]       # (S*M,Nobs)
    sig_draw  = sigma_y[idx_post][:, None]   # (S*M,1)

    eps = rng.standard_normal(size=(S * M, Nobs))
    yrep = mu_draw + sig_draw * eps

    lo50 = onp.percentile(yrep, 25.0, axis=0)
    hi50 = onp.percentile(yrep, 75.0, axis=0)
    lo95 = onp.percentile(yrep,  2.5, axis=0)
    hi95 = onp.percentile(yrep, 97.5, axis=0)

    cov50 = float(onp.mean((y_obs >= lo50) & (y_obs <= hi50)))
    cov95 = float(onp.mean((y_obs >= lo95) & (y_obs <= hi95)))

    return {
        "n_points": Nobs,
        "elpd_per_point": elpd_per_point,
        "rmse": rmse,
        "cov50": cov50,
        "cov95": cov95,
        "rmse_per_freq": rmse_k,
        "n_per_freq": n_k,
        "nrmse_freq": nrmse_freq,
    }


# ==========================================================
# Run one split
# ==========================================================
def run_one_split(split_id, seed_base):
    seed = int(seed_base + 1000 * split_id)

    mask_train_np, mask_test_np, sd_k_train = stratified_point_split(
        mask_obs_np_in=mask_obs_np,
        test_fraction=TEST_FRACTION,
        seed=seed
    )

    V_train = jnp.array(V_filled_np)
    mask_train = jnp.array(mask_train_np)

    master_key = random.PRNGKey(seed)
    kernel = NUTS(
        model_freq_only_notau,
        target_accept_prob=TARGET_ACCEPT,
        init_strategy=init_to_feasible(),
        max_tree_depth=MAX_TREE_DEPTH
    )
    mcmc = MCMC(kernel,
                num_warmup=NUM_WARMUP,
                num_samples=NUM_SAMPLES,
                num_chains=NCHAINS,
                chain_method=CHAIN_METHOD)

    t0 = time.perf_counter()
    mcmc.run(
        master_key,
        f_obs=Exp_Frequency,
        V_obs=V_train,
        mask_obs=mask_train,
        extra_fields=("num_steps", "accept_prob", "diverging", "energy"),
    )
    t1 = time.perf_counter()

    idata = az.from_numpyro(mcmc)
    summ = az.summary(idata, round_to=4)

    print("\n=== Split %d ===" % split_id)
    print("[Split] test_fraction(per-freq)=%.3f | fit_time=%.1fs" % (TEST_FRACTION, (t1 - t0)))
    cols = [c for c in ["mean", "sd", "r_hat", "ess_bulk", "ess_tail"] if c in summ.columns]
    print(summ[cols])

    posterior_samples = mcmc.get_samples()

    train_metrics = compute_elpd_rmse_cov(
        posterior_samples=posterior_samples,
        f_obs=Exp_Frequency,
        V_np=V,
        mask_eval_np=mask_train_np,
        seed=seed + 111,
        sd_k_train=sd_k_train
    )

    test_metrics = compute_elpd_rmse_cov(
        posterior_samples=posterior_samples,
        f_obs=Exp_Frequency,
        V_np=V,
        mask_eval_np=mask_test_np,
        seed=seed + 222,
        sd_k_train=sd_k_train
    )

    elpd_gap = float(train_metrics["elpd_per_point"] - test_metrics["elpd_per_point"])

    out = {
        "split_id": int(split_id),
        "train": train_metrics,
        "test": test_metrics,
        "elpd_gap": elpd_gap,
    }

    print("[TRAIN metrics]")
    print("  n_points = %d" % train_metrics["n_points"])
    print("  ELPD_per_point = %.4f" % train_metrics["elpd_per_point"])
    print("  RMSE_all = %.6e" % train_metrics["rmse"])
    print("  NRMSE_freq = %.4f (standardized by TRAIN SD_k)" % train_metrics["nrmse_freq"])
    print("  Coverage_50 = %.3f" % train_metrics["cov50"])
    print("  Coverage_95 = %.3f" % train_metrics["cov95"])

    print("[TEST metrics]")
    print("  n_points = %d" % test_metrics["n_points"])
    print("  ELPD_per_point = %.4f" % test_metrics["elpd_per_point"])
    print("  RMSE_all = %.6e" % test_metrics["rmse"])
    print("  NRMSE_freq = %.4f (standardized by TRAIN SD_k)" % test_metrics["nrmse_freq"])
    print("  Coverage_50 = %.3f" % test_metrics["cov50"])
    print("  Coverage_95 = %.3f" % test_metrics["cov95"])

    print("[ELPD gap] train_minus_test = %.4f" % elpd_gap)

    print("  RMSE_per_frequency (TEST):")
    for kk in range(K):
        print("    k=%d f=%.4e Hz n_test=%d RMSE_k=%s" % (
            kk, float(freq[kk]), int(test_metrics["n_per_freq"][kk]), str(test_metrics["rmse_per_freq"][kk])
        ))

    return out


# ==========================================================
# Run splits + summarize
# ==========================================================
all_out = []
for s in range(N_SPLITS):
    all_out.append(run_one_split(s, SEED))

def _nanmean(x):
    x = onp.asarray(x, dtype=float)
    return float(onp.nanmean(x)) if x.size else onp.nan

def _nanstd(x):
    x = onp.asarray(x, dtype=float)
    return float(onp.nanstd(x)) if x.size else onp.nan

train_elpd = [o["train"]["elpd_per_point"] for o in all_out]
test_elpd  = [o["test"]["elpd_per_point"] for o in all_out]
gap_elpd   = [o["elpd_gap"] for o in all_out]

print("\n==============================")
print("Across-split summary (TRAIN/TEST)")
print("N_SPLITS = %d" % N_SPLITS)
print("ELPD_train_per_point = %.4f +/- %.4f" % (_nanmean(train_elpd), _nanstd(train_elpd)))
print("ELPD_test_per_point  = %.4f +/- %.4f" % (_nanmean(test_elpd),  _nanstd(test_elpd)))
print("ELPD_gap(train-test) = %.4f +/- %.4f" % (_nanmean(gap_elpd),   _nanstd(gap_elpd)))
print("==============================")

# Keep your previous per-frequency RMSE summary across splits (TEST)
rmse_mat = onp.stack([o["test"]["rmse_per_freq"] for o in all_out], axis=0)
n_mat    = onp.stack([o["test"]["n_per_freq"] for o in all_out], axis=0)

print("\n[Per-frequency RMSE summary across splits] (TEST)")
print("%3s  %12s  %12s  %12s  %10s" % ("k", "freq(Hz)", "RMSE_mean", "RMSE_sd", "avg_n_test"))
for kk in range(K):
    rmse_k_mean = _nanmean(rmse_mat[:, kk])
    rmse_k_sd   = _nanstd(rmse_mat[:, kk])
    avg_nk      = float(onp.mean(n_mat[:, kk]))
    print("%3d  %12.4e  %12.4e  %12.4e  %10.2f" % (kk, float(freq[kk]), rmse_k_mean, rmse_k_sd, avg_nk))