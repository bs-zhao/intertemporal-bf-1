#!/usr/bin/env python
#SBATCH --account=PDE0114
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --job-name=bf8-recovery
#SBATCH --output=slurm-%j.out

"""
Parameter recovery / diagnostics for one already-trained 8-model BayesFlow posterior.

Examples (run from model_fitting_8models/):

    python s2_parameter_recovery.py --model dft_pow_basic_none --mode test
    python s2_parameter_recovery.py --model rudt_asym_decreasing --mode formal

or on OSC:

    sbatch s2_parameter_recovery.py --model rudt_sym_fixed --mode formal

This file NEVER trains a neural network. It imports the exact simulator/prior/screening
source of truth from s1_train_parameter_posterior.py, generates fresh held-out screened
participants, loads posterior_final.keras, and saves numerical recovery diagnostics plus
figures under the corresponding model/mode run directory.
"""

import os
os.environ.setdefault("KERAS_BACKEND", "torch")

import argparse
import gc
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, kstest

# Import BayesFlow before loading the Keras file so custom objects are registered.
import bayesflow as bf
import keras

import s1_train_parameter_posterior as train


TEST_EVAL_CFG = dict(n_test=32, posterior_draws=100, batch_size=16)
FORMAL_EVAL_CFG = dict(n_test=1000, posterior_draws=1000, batch_size=25)
POINT_ESTIMATE = "median"
CENTRAL_CI = 0.90


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parameter recovery for one trained 8-model posterior."
    )
    parser.add_argument("--model", required=True, choices=train.MODEL_NAMES)
    parser.add_argument("--mode", required=True, choices=("test", "formal"))
    parser.add_argument("--n-test", type=int, default=None)
    parser.add_argument("--posterior-draws", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    return parser.parse_args()


def _scalar_vector(x, n):
    x = np.asarray(x)
    x = np.squeeze(x)
    if x.ndim == 0:
        x = x[None]
    if x.ndim != 1 or len(x) != n:
        raise ValueError(f"Expected scalar batch of length {n}; got shape {x.shape}.")
    return x.astype(float)


def _dataset_draw_matrix(x, n_datasets):
    x = np.asarray(x)
    while x.ndim > 2 and x.shape[-1] == 1:
        x = np.squeeze(x, axis=-1)
    if x.ndim == 1:
        if n_datasets != 1:
            raise ValueError(f"Unexpected posterior shape {x.shape} for {n_datasets} datasets.")
        x = x[None, :]
    if x.ndim != 2:
        raise ValueError(f"Expected posterior shape (dataset, draw); got {x.shape}.")
    if x.shape[0] != n_datasets and x.shape[1] == n_datasets:
        x = x.T
    if x.shape[0] != n_datasets:
        raise ValueError(f"Could not identify dataset axis in posterior shape {x.shape}.")
    return x.astype(float)


def _safe_corr(fun, x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    try:
        return float(fun(x, y)[0])
    except Exception:
        return np.nan


def cleanup_memory():
    gc.collect()
    try:
        import torch
        if torch.cuda.device_count() > 0:
            torch.cuda.empty_cache()
    except Exception:
        pass


def resolve_final_model(run_dir):
    preferred = run_dir / "posterior_final.keras"
    if preferred.exists():
        return preferred

    candidates = sorted((run_dir / "checkpoints").glob("*.keras"))
    if not candidates:
        candidates = sorted(run_dir.rglob("*.keras"))
    if not candidates:
        raise FileNotFoundError(
            f"No trained .keras posterior found under {run_dir}. Run s1 first."
        )
    print("WARNING: posterior_final.keras not found; using", candidates[-1])
    return candidates[-1]


def plot_training_history(run_dir, fig_dir):
    path = run_dir / "training_history.csv"
    if not path.exists():
        print("training history not found; skipping loss plot")
        return

    h = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if "loss" in h:
        ax.plot(np.arange(1, len(h) + 1), h["loss"], label="train loss")
    if "val_loss" in h:
        ax.plot(np.arange(1, len(h) + 1), h["val_loss"], label="validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training history")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "training_loss.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    if "learning_rate" in h:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(np.arange(1, len(h) + 1), h["learning_rate"])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning rate")
        ax.set_title("Learning-rate history")
        fig.tight_layout()
        fig.savefig(fig_dir / "learning_rate.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def collect_recovery(approximator, simulator, param_names, cfg):
    rows = []
    corr_sum = np.zeros((len(param_names), len(param_names)), dtype=float)
    abs_corr_sum = np.zeros_like(corr_sum)
    corr_n = np.zeros_like(corr_sum)

    done = 0
    participant_id = 0

    while done < cfg["n_test"]:
        b = min(cfg["batch_size"], cfg["n_test"] - done)
        raw = simulator.sample(b)
        conditions = {"trials": np.asarray(raw["trials"], dtype="float32")}

        samples = approximator.sample(
            conditions=conditions,
            num_samples=cfg["posterior_draws"],
        )

        mats = []
        truths = {}
        for p in param_names:
            truths[p] = _scalar_vector(raw[p], b)
            mats.append(_dataset_draw_matrix(samples[p], b))

        # (batch, draw, parameter)
        posterior_matrix = np.stack(mats, axis=-1)

        for i in range(b):
            # Pairwise posterior dependence for this held-out participant.
            c = np.corrcoef(posterior_matrix[i], rowvar=False)
            finite = np.isfinite(c)
            corr_sum[finite] += c[finite]
            abs_corr_sum[finite] += np.abs(c[finite])
            corr_n[finite] += 1

            for j, p in enumerate(param_names):
                draws = posterior_matrix[i, :, j]
                true = float(truths[p][i])
                point = float(np.median(draws))
                mean = float(np.mean(draws))
                sd = float(np.std(draws, ddof=1))
                alpha = (1.0 - CENTRAL_CI) / 2.0
                lo, hi = np.quantile(draws, [alpha, 1.0 - alpha])

                row = {
                    "participant_id": participant_id + i,
                    "parameter": p,
                    "true": true,
                    "posterior_median": point,
                    "posterior_mean": mean,
                    "posterior_sd": sd,
                    "ci_lower": float(lo),
                    "ci_upper": float(hi),
                    "covered_90": bool(lo <= true <= hi),
                    "rank_u": float(np.mean(draws <= true)),
                }
                if p == "n":
                    row["posterior_median_rounded"] = float(np.clip(np.rint(point), 2, 12))
                    row["n_exact_rounded"] = bool(np.rint(point) == true)
                rows.append(row)

        done += b
        participant_id += b
        print(f"recovery: {done}/{cfg['n_test']} held-out participants")

        del raw, conditions, samples, mats, posterior_matrix
        cleanup_memory()

    mean_corr = np.divide(corr_sum, corr_n, out=np.full_like(corr_sum, np.nan), where=corr_n > 0)
    mean_abs_corr = np.divide(
        abs_corr_sum, corr_n, out=np.full_like(abs_corr_sum, np.nan), where=corr_n > 0
    )
    return pd.DataFrame(rows), mean_corr, mean_abs_corr


def compute_parameter_metrics(long_df, param_names):
    rows = []
    for p in param_names:
        d = long_df[long_df["parameter"] == p].copy()
        true = d["true"].to_numpy(float)
        est = d["posterior_median"].to_numpy(float)
        err = est - true
        true_sd = float(np.std(true, ddof=1)) if len(true) > 1 else np.nan
        rmse = float(np.sqrt(np.mean(err**2)))
        prior_var = float(np.var(true, ddof=1)) if len(true) > 1 else np.nan
        mean_post_var = float(np.mean(d["posterior_sd"].to_numpy(float) ** 2))
        contraction = (
            1.0 - mean_post_var / prior_var
            if np.isfinite(prior_var) and prior_var > 0
            else np.nan
        )

        rank_u = d["rank_u"].to_numpy(float)
        try:
            ks = kstest(rank_u, "uniform")
            ks_stat, ks_p = float(ks.statistic), float(ks.pvalue)
        except Exception:
            ks_stat, ks_p = np.nan, np.nan

        row = {
            "parameter": p,
            "n": len(d),
            "pearson_r": _safe_corr(pearsonr, true, est),
            "spearman_rho": _safe_corr(spearmanr, true, est),
            "rmse": rmse,
            "nrmse_by_true_sd": rmse / true_sd if np.isfinite(true_sd) and true_sd > 0 else np.nan,
            "mae": float(np.mean(np.abs(err))),
            "bias": float(np.mean(err)),
            "coverage_90": float(np.mean(d["covered_90"].astype(float))),
            "mean_ci_width_90": float(np.mean(d["ci_upper"] - d["ci_lower"])),
            "posterior_contraction": contraction,
            "rank_uniform_ks": ks_stat,
            "rank_uniform_p": ks_p,
        }
        if p == "n" and "n_exact_rounded" in d:
            row["rounded_exact_accuracy"] = float(np.mean(d["n_exact_rounded"].astype(float)))
            rounded = d["posterior_median_rounded"].to_numpy(float)
            row["rounded_rmse"] = float(np.sqrt(np.mean((rounded - true) ** 2)))
        rows.append(row)
    return pd.DataFrame(rows)


def plot_recovery_scatter(long_df, param_names, model, fig_dir):
    ncols = 3
    nrows = math.ceil(len(param_names) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.7 * ncols, 4.3 * nrows))
    axes = np.asarray(axes).reshape(-1)

    for ax, p in zip(axes, param_names):
        d = long_df[long_df["parameter"] == p]
        true = d["true"].to_numpy(float)
        est = d["posterior_median"].to_numpy(float)
        ax.scatter(true, est, s=16, alpha=0.55)
        lo = min(float(np.min(true)), float(np.min(est)))
        hi = max(float(np.max(true)), float(np.max(est)))
        ax.plot([lo, hi], [lo, hi], "--", linewidth=1)
        r = _safe_corr(pearsonr, true, est)
        ax.set_xlabel(f"True {p}")
        ax.set_ylabel(f"Posterior median {p}")
        ax.set_title(f"{p}: r={r:.3f}" if np.isfinite(r) else p)

    for ax in axes[len(param_names):]:
        ax.axis("off")
    fig.suptitle(f"Parameter recovery — {model}", y=1.01)
    fig.tight_layout()
    fig.savefig(fig_dir / "parameter_recovery_scatter.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_error_vs_truth(long_df, param_names, model, fig_dir):
    ncols = 3
    nrows = math.ceil(len(param_names) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.7 * ncols, 4.0 * nrows))
    axes = np.asarray(axes).reshape(-1)

    for ax, p in zip(axes, param_names):
        d = long_df[long_df["parameter"] == p]
        true = d["true"].to_numpy(float)
        err = d["posterior_median"].to_numpy(float) - true
        ax.scatter(true, err, s=16, alpha=0.55)
        ax.axhline(0.0, linestyle="--", linewidth=1)
        ax.set_xlabel(f"True {p}")
        ax.set_ylabel("Posterior median - true")
        ax.set_title(p)

    for ax in axes[len(param_names):]:
        ax.axis("off")
    fig.suptitle(f"Recovery error vs truth — {model}", y=1.01)
    fig.tight_layout()
    fig.savefig(fig_dir / "parameter_recovery_error.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_rank_calibration(long_df, param_names, model, fig_dir):
    ncols = 3
    nrows = math.ceil(len(param_names) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.7 * ncols, 3.7 * nrows))
    axes = np.asarray(axes).reshape(-1)

    for ax, p in zip(axes, param_names):
        rank_u = long_df.loc[long_df["parameter"] == p, "rank_u"].to_numpy(float)
        ax.hist(rank_u, bins=np.linspace(0, 1, 11), density=True, alpha=0.7)
        ax.axhline(1.0, linestyle="--", linewidth=1)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Posterior CDF at true value")
        ax.set_ylabel("Density")
        ax.set_title(p)

    for ax in axes[len(param_names):]:
        ax.axis("off")
    fig.suptitle(f"Posterior rank calibration — {model}", y=1.01)
    fig.tight_layout()
    fig.savefig(fig_dir / "posterior_rank_calibration.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_metric_bars(metrics, fig_dir):
    fig, ax = plt.subplots(figsize=(max(7, 0.9 * len(metrics)), 4.5))
    ax.bar(metrics["parameter"], metrics["coverage_90"])
    ax.axhline(CENTRAL_CI, linestyle="--", linewidth=1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Empirical coverage")
    ax.set_title("90% posterior interval coverage")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(fig_dir / "coverage_90.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(max(7, 0.9 * len(metrics)), 4.5))
    ax.bar(metrics["parameter"], metrics["posterior_contraction"])
    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_ylabel("1 - mean posterior variance / prior variance")
    ax.set_title("Posterior contraction")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(fig_dir / "posterior_contraction.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_corr_heatmap(matrix, param_names, title, path, signed):
    fig, ax = plt.subplots(
        figsize=(max(6.2, 0.82 * len(param_names)), max(5.4, 0.72 * len(param_names)))
    )
    if signed:
        im = ax.imshow(matrix, vmin=-1, vmax=1)
    else:
        im = ax.imshow(matrix, vmin=0, vmax=1)
    ax.set_xticks(range(len(param_names)), param_names, rotation=45, ha="right")
    ax.set_yticks(range(len(param_names)), param_names)
    ax.set_title(title)
    for i in range(len(param_names)):
        for j in range(len(param_names)):
            if np.isfinite(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_n_recovery(long_df, fig_dir):
    d = long_df[long_df["parameter"] == "n"].copy()
    if d.empty:
        return

    true_vals = np.arange(train.RUDT_N_RANGE[0], train.RUDT_N_RANGE[1] + 1)
    pred = d["posterior_median_rounded"].to_numpy(int)
    true = d["true"].to_numpy(int)
    matrix = np.zeros((len(true_vals), len(true_vals)), dtype=int)
    for t, p in zip(true, pred):
        if t in true_vals and p in true_vals:
            matrix[t - true_vals[0], p - true_vals[0]] += 1

    denom = matrix.sum(axis=1, keepdims=True)
    prop = np.divide(matrix, denom, out=np.zeros_like(matrix, dtype=float), where=denom > 0)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(prop, vmin=0, vmax=1)
    ax.set_xticks(range(len(true_vals)), true_vals)
    ax.set_yticks(range(len(true_vals)), true_vals)
    ax.set_xlabel("Rounded posterior median n")
    ax.set_ylabel("True n")
    ax.set_title("Discrete n recovery")
    fig.colorbar(im, ax=ax, label="Row proportion")
    fig.tight_layout()
    fig.savefig(fig_dir / "n_recovery_confusion.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    train.check_environment()
    registry = train.load_model_registry()
    design, ter_pop = train.load_demurie_design_and_ter()
    param_names = registry[args.model][1]

    cfg = (TEST_EVAL_CFG.copy() if args.mode == "test" else FORMAL_EVAL_CFG.copy())
    if args.n_test is not None:
        cfg["n_test"] = args.n_test
    if args.posterior_draws is not None:
        cfg["posterior_draws"] = args.posterior_draws
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size

    run_dir = train.get_run_dir(args.model, args.mode)
    diag_dir = run_dir / "diagnostics"
    fig_dir = diag_dir / "figures"
    diag_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    model_path = resolve_final_model(run_dir)
    print("loading:", model_path)
    approximator = keras.saving.load_model(model_path)
    print("loaded class:", type(approximator).__name__)

    model_idx = train.MODEL_NAMES.index(args.model)
    simulator, redraw_stats = train.make_parameter_simulator(
        model=args.model,
        registry=registry,
        design_df=design,
        ter_pop=ter_pop,
        seed=train.BASE_SEED + 100_000 + 1000 * model_idx,
    )

    print("evaluation config:", cfg)
    print("parameters:", param_names)
    plot_training_history(run_dir, fig_dir)

    long_df, mean_corr, mean_abs_corr = collect_recovery(
        approximator=approximator,
        simulator=simulator,
        param_names=param_names,
        cfg=cfg,
    )
    metrics = compute_parameter_metrics(long_df, param_names)

    long_df.to_csv(diag_dir / "parameter_recovery_long.csv", index=False)
    metrics.to_csv(diag_dir / "parameter_recovery_metrics.csv", index=False)
    pd.DataFrame(mean_corr, index=param_names, columns=param_names).to_csv(
        diag_dir / "posterior_correlation_mean_signed.csv"
    )
    pd.DataFrame(mean_abs_corr, index=param_names, columns=param_names).to_csv(
        diag_dir / "posterior_correlation_mean_abs.csv"
    )

    plot_recovery_scatter(long_df, param_names, args.model, fig_dir)
    plot_error_vs_truth(long_df, param_names, args.model, fig_dir)
    plot_rank_calibration(long_df, param_names, args.model, fig_dir)
    plot_metric_bars(metrics, fig_dir)
    plot_corr_heatmap(
        mean_corr,
        param_names,
        "Mean signed within-dataset posterior correlation",
        fig_dir / "posterior_correlation_mean_signed.png",
        signed=True,
    )
    plot_corr_heatmap(
        mean_abs_corr,
        param_names,
        "Mean absolute within-dataset posterior correlation",
        fig_dir / "posterior_correlation_mean_abs.png",
        signed=False,
    )
    if "n" in param_names:
        plot_n_recovery(long_df, fig_dir)

    summary = {
        "model": args.model,
        "mode": args.mode,
        "parameter_names": list(param_names),
        "evaluation_config": cfg,
        "trained_model": str(model_path),
        "diagnostics_dir": str(diag_dir),
        "fresh_test_redraw_stats": redraw_stats,
        "note_on_n": (
            "n is discrete in the simulator but is represented by the same continuous "
            "CouplingFlow requested for all models; rounded recovery metrics are reported."
            if "n" in param_names else None
        ),
    }
    (diag_dir / "diagnostics_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 80)
    print("PARAMETER RECOVERY COMPLETE")
    print("=" * 80)
    print(metrics.to_string(index=False))
    print("\noutputs:", diag_dir)
    print("fresh held-out redraw stats:", redraw_stats)

    del approximator
    keras.backend.clear_session()
    cleanup_memory()


if __name__ == "__main__":
    main()