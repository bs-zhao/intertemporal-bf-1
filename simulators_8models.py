"""
Self-contained simulators, priors, and Demurie design for the 8-model BayesFlow project.

This file deliberately does NOT import anything from the repository's main simulator code.
It reimplements the four DFT and four RUDT models used in demurie2013_sim.ipynb:

    dft_pow_basic_none
    dft_pow_basic_fixed
    dft_pow_basic_driftsign
    dft_pow_modified_none
    rudt_sym_fixed
    rudt_sym_decreasing
    rudt_asym_fixed
    rudt_asym_decreasing

The implementation follows the formulas and project choices in the current main-branch
simulator notebook/code supplied by the user.

The Demurie task design is reconstructed directly from the documented full factorial:
SS = EUR 0/5/10/20/30 today; LL = EUR 30 after 0/1/2/7/14 days; 4 repetitions;
the identical EUR30-today vs EUR30-today problem is removed, yielding 96 trials.

The original notebook computed the Ter prior from Demurie_2013.csv. That CSV is not present
in the supplied main-branch ZIP, so this independent folder freezes the values printed by
that notebook:
    Ter ~ truncated Normal(mean=0.694, sd=0.252, lower=0.2, upper=1.31)
This removes the external data-file dependency while preserving the reported prior.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm, truncnorm
from ssms.basic_simulators.simulator import simulator as ssms_simulator


# =============================================================================
# Public model registry / project constants
# =============================================================================

MODEL_NAMES = (
    "dft_pow_basic_none",
    "dft_pow_basic_fixed",
    "dft_pow_basic_driftsign",
    "dft_pow_modified_none",
    "rudt_sym_fixed",
    "rudt_sym_decreasing",
    "rudt_asym_fixed",
    "rudt_asym_decreasing",
)

PARAM_NAMES = {
    "dft_pow_basic_none": ("w", "theta_star", "alpha", "beta", "Ter"),
    "dft_pow_basic_fixed": ("w", "theta_star", "alpha", "beta", "Ter", "z_star"),
    "dft_pow_basic_driftsign": ("w", "theta_star", "alpha", "beta", "Ter", "z_star"),
    "dft_pow_modified_none": ("w", "theta_star", "alpha", "beta", "Ter"),
    "rudt_sym_fixed": ("alpha", "beta", "s", "c", "theta", "Ter", "delta"),
    "rudt_sym_decreasing": ("alpha", "beta", "s", "c", "theta", "Ter", "delta", "n"),
    "rudt_asym_fixed": ("alpha", "beta", "s", "c", "theta", "Ter", "delta_S", "delta_L"),
    "rudt_asym_decreasing": (
        "alpha", "beta", "s", "c", "theta", "Ter", "delta_S", "delta_L", "n"
    ),
}

ATTRS = ("ss", "delay_ss", "ll", "delay_ll")
DESIGN_SCALE = 80.0
TRIAL_FEATURE_NAMES = (
    "ss_scaled",
    "delay_ss_scaled",
    "ll_scaled",
    "delay_ll_scaled",
    "choice_signed",
    "log1p_rt",
)

MAX_REDRAW = 200
SCREEN_MAX_MEAN_RT = 10.0
# Executable code in the supplied notebook uses .025/.975.
SCREEN_P_LL = (0.025, 0.975)

# Frozen from printed output of the supplied demurie2013_sim.ipynb.
TER_POP = (0.694, 0.252, 0.2, 1.31)

# DFT Table 8 project priors: (mean, sd, lower, upper)
DFT_POP = {
    "w": (0.54, 0.29, 0.05, 0.95),
    "theta_star": (1.76, 0.33, 0.01, 100.0),
    "alpha": (0.57, 0.35, 0.01, 1.0),
    "beta": (0.45, 0.38, 0.01, 1.0),
    "Ter": TER_POP,
}
DFT_Z_STAR = (0.34, 0.27)  # |z*|, truncated to [0, .9*theta_star]

# RUDT Table 10 project priors
RUDT_POP_NORMAL = {
    "alpha": (0.64, 0.35, 0.01, 1.0),
    "beta": (0.63, 0.37, 0.01, 1.0),
    "theta": (1.10, 0.47, 0.01, np.inf),
    "Ter": TER_POP,
}
RUDT_POP_LOGNORMAL = {
    "s": (0.62, 8.14, 0.0, 100.0),
    "c": (0.08, 0.28, 0.0, 1.0),
    "delta": (0.09, 0.40, 0.0, None),
}
RUDT_N_RANGE = (2, 12)
P_STOP_MIN = 1e-6


# =============================================================================
# Demurie 96-trial design (self-contained; no CSV required)
# =============================================================================

def build_demurie_design() -> pd.DataFrame:
    """Reconstruct the exact 24-problem x 4-repetition design and fixed shuffle."""
    rows = []
    for ss in (0, 5, 10, 20, 30):
        for delay_ll in (0, 1, 2, 7, 14):
            # LL is always EUR 30; SS is always today.
            if ss == 30 and delay_ll == 0:
                # Identical EUR30 today vs EUR30 today problem is undefined in both families.
                continue
            for rep in range(1, 5):
                rows.append(
                    {
                        "ss": ss,
                        "delay_ss": 0,
                        "ll": 30,
                        "delay_ll": delay_ll,
                        "rep": rep,
                    }
                )

    design = pd.DataFrame(rows)
    design = design.sample(frac=1, random_state=42).reset_index(drop=True)
    if len(design) != 96:
        raise RuntimeError(f"Expected 96 Demurie trials, got {len(design)}")
    return design


DEMURIE_DESIGN = build_demurie_design()


# =============================================================================
# Shared simulator helpers
# =============================================================================

def require_positive(params, names):
    for name in names:
        if not params[name] > 0:
            raise ValueError(f"parameter {name!r} must be > 0, got {params[name]!r}")


def validate_design(design_df, allow_ties=False):
    ss = np.asarray(design_df["ss"], dtype=float)
    ll = np.asarray(design_df["ll"], dtype=float)
    delay_ss = np.asarray(design_df["delay_ss"], dtype=float)
    delay_ll = np.asarray(design_df["delay_ll"], dtype=float)

    if allow_ties:
        if not np.all(ss >= 0):
            raise ValueError("design must satisfy ss >= 0 on every trial")
        if not np.all(ll >= ss):
            raise ValueError("design must satisfy ll >= ss on every trial")
        if not np.all(delay_ll >= delay_ss):
            raise ValueError("design must satisfy delay_ll >= delay_ss on every trial")
        if not np.all(delay_ss >= 0):
            raise ValueError("design must satisfy delay_ss >= 0 on every trial")
        if np.any((ll == ss) & (delay_ll == delay_ss)):
            raise ValueError("design has trials with two identical options")
        return ss, delay_ss, ll, delay_ll

    if not np.all(ss > 0):
        raise ValueError("design must satisfy ss > 0 on every trial")
    if not np.all(ll > ss):
        raise ValueError("design must satisfy ll > ss on every trial")
    if not np.all(delay_ll > delay_ss):
        raise ValueError("design must satisfy delay_ll > delay_ss on every trial")
    if not np.all(delay_ss >= 0):
        raise ValueError("design must satisfy delay_ss >= 0 on every trial")
    return ss, delay_ss, ll, delay_ll


def simulate_ddm(design_df, v, a, z, t, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    n = len(design_df)
    cols = [np.broadcast_to(np.asarray(x, dtype=float), (n,)) for x in (v, a, z, t)]
    theta = np.column_stack(cols)

    seed = int(rng.integers(0, 2**31 - 1))
    out = ssms_simulator(theta=theta, model="ddm", n_samples=1, random_state=seed)

    result_df = design_df.copy()
    result_df["choice"] = (out["choices"][:, 0] == 1).astype(int)
    result_df["rt"] = out["rts"][:, 0]
    return result_df


# =============================================================================
# DFT: four selected models
# =============================================================================

def dft_probs(design_df, params, *, eval_mode, bias_mode):
    """Power-time DFT for the selected basic/modified and none/fixed/driftsign variants."""
    names = PARAM_NAMES[
        {
            ("basic", "none"): "dft_pow_basic_none",
            ("basic", "fixed"): "dft_pow_basic_fixed",
            ("basic", "driftsign"): "dft_pow_basic_driftsign",
            ("modified", "none"): "dft_pow_modified_none",
        }[(eval_mode, bias_mode)]
    ]

    require_positive(params, [n for n in names if n != "z_star"])
    w, theta_star, alpha, beta, Ter = (
        params[k] for k in ("w", "theta_star", "alpha", "beta", "Ter")
    )
    if not 0 < w < 1:
        raise ValueError(f"w must be in (0, 1), got {w!r}")

    ss, delay_ss, ll, delay_ll = validate_design(design_df, allow_ties=True)

    du = ll**alpha - ss**alpha
    dp = delay_ll**beta - delay_ss**beta

    if eval_mode == "basic":
        A, B = du, dp
    elif eval_mode == "modified":
        A = (ll - ss) ** alpha
        B = (delay_ll - delay_ss) ** beta
    else:
        raise ValueError(f"Unsupported eval_mode: {eval_mode}")

    d = w * A - (1 - w) * B
    sigma = np.sqrt(w * (1 - w) * (A + B) ** 2)

    if bias_mode == "none":
        z_star_eff = np.zeros_like(d)
    elif bias_mode == "fixed":
        z_star = params["z_star"]
        if not abs(z_star) < theta_star:
            raise ValueError(
                f"|z_star| ({abs(z_star)!r}) must be < theta_star ({theta_star!r})"
            )
        z_star_eff = np.full_like(d, z_star)
    elif bias_mode == "driftsign":
        z_star = params["z_star"]
        if not 0 <= z_star < theta_star:
            raise ValueError(f"z_star ({z_star!r}) must be in [0, theta_star)")
        z_star_eff = z_star * np.sign(d)
    else:
        raise ValueError(f"Unsupported bias_mode: {bias_mode}")

    v = d / sigma
    a = np.full_like(v, theta_star)
    z = 0.5 + z_star_eff / (2 * theta_star)
    t = np.full_like(v, Ter)

    return pd.DataFrame({"d": d, "sigma": sigma, "v": v, "a": a, "z": z, "t": t})


def _simulate_dft(design_df, params, *, eval_mode, bias_mode, rng=None):
    theta = dft_probs(design_df, params, eval_mode=eval_mode, bias_mode=bias_mode)
    return simulate_ddm(
        design_df,
        v=theta["v"],
        a=theta["a"],
        z=theta["z"],
        t=theta["t"],
        rng=rng,
    )


def simulate_dft_pow_basic_none(design_df, params, rng=None):
    return _simulate_dft(design_df, params, eval_mode="basic", bias_mode="none", rng=rng)


def simulate_dft_pow_basic_fixed(design_df, params, rng=None):
    return _simulate_dft(design_df, params, eval_mode="basic", bias_mode="fixed", rng=rng)


def simulate_dft_pow_basic_driftsign(design_df, params, rng=None):
    return _simulate_dft(
        design_df, params, eval_mode="basic", bias_mode="driftsign", rng=rng
    )


def simulate_dft_pow_modified_none(design_df, params, rng=None):
    return _simulate_dft(design_df, params, eval_mode="modified", bias_mode="none", rng=rng)


# =============================================================================
# RUDT: four selected models
# =============================================================================

def rudt_d_sigma(design_df, params):
    ss, delay_ss, ll, delay_ll = validate_design(design_df, allow_ties=True)
    alpha, beta, s, c = params["alpha"], params["beta"], params["s"], params["c"]

    delta_M = ll**alpha - ss**alpha
    delta_T = s * (delay_ll**beta - delay_ss**beta)

    d = delta_M - delta_T
    sigma = np.sqrt(c * (delta_M + delta_T))
    return d, sigma


def _rudt_thresholds(params, symmetry):
    if symmetry == "sym":
        return params["delta"], params["delta"]
    if symmetry == "asym":
        return params["delta_S"], params["delta_L"]
    raise ValueError(f"symmetry must be 'sym' or 'asym', got {symmetry!r}")


def _rudt_param_names(symmetry, trend):
    if symmetry == "sym":
        names = ["alpha", "beta", "s", "c", "theta", "Ter", "delta"]
    elif symmetry == "asym":
        names = ["alpha", "beta", "s", "c", "theta", "Ter", "delta_S", "delta_L"]
    else:
        raise ValueError(symmetry)
    if trend == "decreasing":
        names.append("n")
    return tuple(names)


def simulate_rudt_fixed(design_df, params, *, symmetry, rng=None, p_stop_min=P_STOP_MIN):
    if rng is None:
        rng = np.random.default_rng()

    require_positive(params, _rudt_param_names(symmetry, "fixed"))
    d, sigma = rudt_d_sigma(design_df, params)
    delta_S, delta_L = _rudt_thresholds(params, symmetry)

    P_L = norm.sf((delta_L - d) / sigma)
    P_S = norm.cdf((-delta_S - d) / sigma)
    p_stop = P_L + P_S

    if np.any(p_stop < p_stop_min):
        n_bad = int(np.sum(p_stop < p_stop_min))
        worst = float(np.min(p_stop))
        raise ValueError(
            f"{n_bad}/{len(p_stop)} trials have p_stop < {p_stop_min:g} "
            f"(min {worst:g})"
        )

    theta, Ter = params["theta"], params["Ter"]
    n_samples = rng.geometric(p_stop)
    choices = (rng.random(len(p_stop)) < P_L / p_stop).astype(int)
    rts = rng.gamma(shape=2.0 * n_samples, scale=theta) + Ter

    result_df = design_df.copy()
    result_df["choice"] = choices
    result_df["rt"] = rts
    result_df["n_samples"] = n_samples
    return result_df


def simulate_rudt_decreasing(design_df, params, *, symmetry, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    require_positive(params, _rudt_param_names(symmetry, "decreasing"))

    n = params["n"]
    if not (float(n).is_integer() and n >= 1):
        raise ValueError(f"n must be a positive integer, got {n!r}")
    n = int(n)

    d, sigma = rudt_d_sigma(design_df, params)
    delta_S0, delta_L0 = _rudt_thresholds(params, symmetry)
    theta, Ter = params["theta"], params["Ter"]

    N = len(d)
    choice = np.full(N, -1, dtype=np.int8)
    rt_samples = np.zeros(N)
    n_samples = np.zeros(N, dtype=np.int64)
    done = np.zeros(N, dtype=bool)

    for k in range(1, n + 1):
        frac = 0.0 if n == 1 else (n - k) / (n - 1)
        delta_S_k = delta_S0 * frac
        delta_L_k = delta_L0 * frac

        pending = ~done
        if not pending.any():
            break

        eval_k = rng.normal(loc=d[pending], scale=sigma[pending])
        rt_samples[pending] += rng.gamma(shape=2.0, scale=theta, size=pending.sum())
        n_samples[pending] = k

        idx = np.flatnonzero(pending)
        decide_L = eval_k > delta_L_k
        decide_S = eval_k < -delta_S_k
        choice[idx[decide_L]] = 1
        choice[idx[decide_S]] = 0
        done[idx[decide_L | decide_S]] = True

    if not done.all():
        raise RuntimeError(
            f"{int((~done).sum())} trial(s) failed to decide by sample n={n}"
        )

    result_df = design_df.copy()
    result_df["choice"] = choice
    result_df["rt"] = rt_samples + Ter
    result_df["n_samples"] = n_samples
    return result_df


def simulate_rudt_sym_fixed(design_df, params, rng=None):
    return simulate_rudt_fixed(design_df, params, symmetry="sym", rng=rng)


def simulate_rudt_sym_decreasing(design_df, params, rng=None):
    return simulate_rudt_decreasing(design_df, params, symmetry="sym", rng=rng)


def simulate_rudt_asym_fixed(design_df, params, rng=None):
    return simulate_rudt_fixed(design_df, params, symmetry="asym", rng=rng)


def simulate_rudt_asym_decreasing(design_df, params, rng=None):
    return simulate_rudt_decreasing(design_df, params, symmetry="asym", rng=rng)


SIMULATORS = {
    "dft_pow_basic_none": simulate_dft_pow_basic_none,
    "dft_pow_basic_fixed": simulate_dft_pow_basic_fixed,
    "dft_pow_basic_driftsign": simulate_dft_pow_basic_driftsign,
    "dft_pow_modified_none": simulate_dft_pow_modified_none,
    "rudt_sym_fixed": simulate_rudt_sym_fixed,
    "rudt_sym_decreasing": simulate_rudt_sym_decreasing,
    "rudt_asym_fixed": simulate_rudt_asym_fixed,
    "rudt_asym_decreasing": simulate_rudt_asym_decreasing,
}


# =============================================================================
# Priors: same project choices as supplied demurie2013_sim.ipynb
# =============================================================================

def _truncnorm(mean, sd, lo, hi, size, rng):
    a = (np.asarray(lo) - mean) / sd
    b = (np.asarray(hi) - mean) / sd
    return truncnorm.rvs(a, b, loc=mean, scale=sd, size=size, random_state=rng)


def _trunclognorm(median, mean, lo, hi, size, rng):
    mu = np.log(median)
    sigma = np.sqrt(2 * np.log(mean / median))
    with np.errstate(divide="ignore"):
        p_lo = norm.cdf((np.log(lo) - mu) / sigma)
        p_hi = norm.cdf((np.log(hi) - mu) / sigma)
    return np.exp(mu + sigma * norm.ppf(rng.uniform(p_lo, p_hi, size)))


def rudt_delta_max(design_df, alpha, beta, s):
    x_max = design_df[["ss", "ll"]].to_numpy().max()
    t_max = design_df[["delay_ss", "delay_ll"]].to_numpy().max()
    return np.maximum(x_max**alpha, s * t_max**beta)


def sample_dft_params(model, size, rng):
    names = PARAM_NAMES[model]
    draws = {k: _truncnorm(*DFT_POP[k], size, rng) for k in DFT_POP}

    bias_mode = model.split("_")[-1]
    if bias_mode != "none":
        z_mag = _truncnorm(
            DFT_Z_STAR[0],
            DFT_Z_STAR[1],
            0.0,
            0.9 * draws["theta_star"],
            size,
            rng,
        )
        if bias_mode == "fixed":
            z_mag = z_mag * rng.choice([-1.0, 1.0], size=size)
        draws["z_star"] = z_mag

    return pd.DataFrame(draws)[list(names)]


def sample_rudt_params(model, size, design_df, rng):
    names = PARAM_NAMES[model]
    draws = {k: _truncnorm(*RUDT_POP_NORMAL[k], size, rng) for k in RUDT_POP_NORMAL}

    for k in ("s", "c"):
        draws[k] = _trunclognorm(*RUDT_POP_LOGNORMAL[k], size, rng)

    delta_max = rudt_delta_max(design_df, draws["alpha"], draws["beta"], draws["s"])
    median, mean, lo, _ = RUDT_POP_LOGNORMAL["delta"]
    delta_names = ("delta",) if "delta" in names else ("delta_S", "delta_L")
    for k in delta_names:
        draws[k] = _trunclognorm(median, mean, lo, delta_max, size, rng)

    if "n" in names:
        draws["n"] = rng.integers(RUDT_N_RANGE[0], RUDT_N_RANGE[1] + 1, size=size)

    return pd.DataFrame(draws)[list(names)]


def sample_population_params(model, size, design_df=None, rng=None):
    if model not in MODEL_NAMES:
        raise ValueError(f"Unknown model {model!r}. Choices: {MODEL_NAMES}")
    if design_df is None:
        design_df = DEMURIE_DESIGN
    if rng is None:
        rng = np.random.default_rng()

    if model.startswith("dft_"):
        return sample_dft_params(model, size, rng)
    return sample_rudt_params(model, size, design_df, rng)


# =============================================================================
# Screening and BayesFlow trial encoding
# =============================================================================

def screen_simulated_subject(sim_df):
    if sim_df["rt"].mean() > SCREEN_MAX_MEAN_RT:
        return "slow"
    p_ll = sim_df["choice"].mean()
    if not SCREEN_P_LL[0] <= p_ll <= SCREEN_P_LL[1]:
        return "one-sided"
    return None


def encode_trials(sim_df):
    choice = np.asarray(sim_df["choice"], dtype=float)
    return np.column_stack(
        [
            np.asarray(sim_df["ss"], dtype=float) / DESIGN_SCALE,
            np.asarray(sim_df["delay_ss"], dtype=float) / DESIGN_SCALE,
            np.asarray(sim_df["ll"], dtype=float) / DESIGN_SCALE,
            np.asarray(sim_df["delay_ll"], dtype=float) / DESIGN_SCALE,
            2.0 * choice - 1.0,
            np.log1p(np.asarray(sim_df["rt"], dtype=float)),
        ]
    ).astype("float32")


def draw_screened_subject(model, rng=None, design_df=None, redraw_stats=None):
    if model not in MODEL_NAMES:
        raise ValueError(f"Unknown model {model!r}. Choices: {MODEL_NAMES}")
    if rng is None:
        rng = np.random.default_rng()
    if design_df is None:
        design_df = DEMURIE_DESIGN
    if redraw_stats is None:
        redraw_stats = {"p_stop": 0, "slow": 0, "one-sided": 0}

    simulate = SIMULATORS[model]
    names = PARAM_NAMES[model]

    for _ in range(MAX_REDRAW + 1):
        params = sample_population_params(model, 1, design_df, rng).iloc[0]
        param_dict = params.to_dict()
        if "n" in param_dict:
            param_dict["n"] = int(param_dict["n"])

        try:
            sim_df = simulate(design_df, param_dict, rng=rng)
            reason = screen_simulated_subject(sim_df)
        except ValueError:
            # Matches the original population simulator's treatment of fixed-RUDT
            # p_stop failures as redraws.
            reason = "p_stop"

        if reason is None:
            return params, sim_df
        redraw_stats[reason] += 1

    raise RuntimeError(
        f"{model}: no accepted parameter/data draw after {MAX_REDRAW} redraws"
    )


def make_bayesflow_parameter_simulator(bf, model, seed):
    """Return bf.make_simulator(...) plus mutable redraw counters."""
    if model not in MODEL_NAMES:
        raise ValueError(f"Unknown model {model!r}. Choices: {MODEL_NAMES}")

    rng = np.random.default_rng(seed)
    names = PARAM_NAMES[model]
    redraw_stats = {"p_stop": 0, "slow": 0, "one-sided": 0, "accepted": 0}

    def simulate_one():
        params, sim_df = draw_screened_subject(
            model=model,
            rng=rng,
            design_df=DEMURIE_DESIGN,
            redraw_stats=redraw_stats,
        )
        redraw_stats["accepted"] += 1

        out = {name: float(params[name]) for name in names}
        out["trials"] = encode_trials(sim_df)
        return out

    return bf.make_simulator(simulate_one), redraw_stats


# =============================================================================
# Small validation helper
# =============================================================================

def describe_model(model):
    if model not in MODEL_NAMES:
        raise ValueError(model)
    return {
        "model": model,
        "parameter_names": list(PARAM_NAMES[model]),
        "n_trials": len(DEMURIE_DESIGN),
        "ter_pop": list(TER_POP),
        "screen_max_mean_rt": SCREEN_MAX_MEAN_RT,
        "screen_p_ll": list(SCREEN_P_LL),
    }
