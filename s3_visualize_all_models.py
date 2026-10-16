#!/usr/bin/env python
import argparse, json, math, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DFT_MODELS = [
    "dft_pow_basic_none",
    "dft_pow_basic_fixed",
    "dft_pow_basic_driftsign",
    "dft_pow_modified_none",
]
RUDT_MODELS = [
    "rudt_sym_fixed",
    "rudt_sym_decreasing",
    "rudt_asym_fixed",
    "rudt_asym_decreasing",
]
MODEL_FAMILIES = {"DFT": DFT_MODELS, "RUDT": RUDT_MODELS}
ALL_MODELS = DFT_MODELS + RUDT_MODELS

PREFERRED_METRICS = [
    "pearson_r", "spearman_rho", "rmse", "nrmse_by_true_sd",
    "mae", "bias", "coverage_90", "mean_ci_width_90", "posterior_contraction",
    "rounded_exact_accuracy", "rounded_rmse",
]
HEATMAP_METRICS = ["pearson_r", "rmse", "coverage_90", "posterior_contraction", "nrmse_by_true_sd", "bias", "mean_ci_width_90", "rank_uniform_ks"]
LOWER_IS_BETTER = {"rmse", "nrmse_by_true_sd", "mae", "mean_ci_width_90", "rounded_rmse"}

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)

def nice_model_name(model):
    return model.replace("_", "\n")

def load_csv_if_exists(path):
    p = Path(path)
    return pd.read_csv(p) if p.exists() else None

def load_json_if_exists(path):
    p = Path(path)
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def gather_model_outputs(output_root, model, mode):
    base = Path(output_root) / model / mode
    diag = base / "diagnostics"
    return {
        "model": model,
        "family": "DFT" if model in DFT_MODELS else "RUDT",
        "base_dir": base,
        "diag_dir": diag,
        "history": load_csv_if_exists(base / "training_history.csv"),
        "metrics": load_csv_if_exists(diag / "parameter_recovery_metrics.csv"),
        "long": load_csv_if_exists(diag / "parameter_recovery_long.csv"),
        "summary": load_json_if_exists(diag / "diagnostics_summary.json"),
        "corr_signed": load_csv_if_exists(diag / "posterior_correlation_mean_signed.csv"),
        "corr_abs": load_csv_if_exists(diag / "posterior_correlation_mean_abs.csv"),
    }


def diagnostics_complete(output_root, model, mode):
    diag = Path(output_root) / model / mode / "diagnostics"
    required = [
        diag / "parameter_recovery_metrics.csv",
        diag / "parameter_recovery_long.csv",
        diag / "posterior_correlation_mean_signed.csv",
        diag / "posterior_correlation_mean_abs.csv",
    ]
    return all(p.exists() for p in required)


def ensure_trained_models_exist(output_root, mode):
    missing = []
    for model in ALL_MODELS:
        p = Path(output_root) / model / mode / "posterior_final.keras"
        if not p.exists():
            missing.append(str(p))
    if missing:
        raise FileNotFoundError(
            "Missing trained posterior(s):\n  " + "\n  ".join(missing)
        )


def run_missing_recoveries(output_root, mode, n_test, posterior_draws, batch_size, force=False):
    ensure_trained_models_exist(output_root, mode)

    for i, model in enumerate(ALL_MODELS, start=1):
        if diagnostics_complete(output_root, model, mode) and not force:
            print(f"[{i}/8] {model}: diagnostics already exist; reusing them.")
            continue

        cmd = [
            sys.executable,
            "s2_parameter_recovery.py",
            "--model", model,
            "--mode", mode,
            "--n-test", str(n_test),
            "--posterior-draws", str(posterior_draws),
            "--batch-size", str(batch_size),
        ]
        print("\n" + "=" * 80)
        print(f"[{i}/8] PARAMETER RECOVERY: {model}")
        print(" ".join(cmd))
        print("=" * 80, flush=True)
        subprocess.run(cmd, check=True)

        if not diagnostics_complete(output_root, model, mode):
            raise RuntimeError(
                f"{model}: s2 finished but expected diagnostics files are still missing."
            )


def concat_metrics(model_outputs):
    frames = []
    for d in model_outputs:
        if d["metrics"] is not None:
            df = d["metrics"].copy()
            df.insert(0, "model", d["model"])
            df.insert(1, "family", d["family"])
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def concat_long(model_outputs):
    frames = []
    for d in model_outputs:
        if d["long"] is not None:
            df = d["long"].copy()
            df.insert(0, "model", d["model"])
            df.insert(1, "family", d["family"])
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def concat_history(model_outputs):
    frames = []
    for d in model_outputs:
        if d["history"] is not None:
            df = d["history"].copy()
            df.insert(0, "epoch", np.arange(1, len(df) + 1))
            df.insert(0, "model", d["model"])
            df.insert(1, "family", d["family"])
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def save_tables(metrics_df, long_df, history_df, out_dir):
    if not metrics_df.empty:
        metrics_df.to_csv(out_dir / "all_models_parameter_recovery_metrics.csv", index=False)
    if not long_df.empty:
        long_df.to_csv(out_dir / "all_models_parameter_recovery_long.csv", index=False)
    if not history_df.empty:
        history_df.to_csv(out_dir / "all_models_training_history.csv", index=False)

def discover_metric_columns(metrics_df):
    return [c for c in PREFERRED_METRICS if c in metrics_df.columns]

def plot_training_curves(history_df, family, out_dir):
    if history_df.empty or "loss" not in history_df.columns:
        return
    df = history_df[history_df["family"] == family]
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for model in MODEL_FAMILIES[family]:
        d = df[df["model"] == model]
        if d.empty:
            continue
        ax.plot(d["epoch"], d["loss"], label=f"{model} train")
        if "val_loss" in d.columns:
            ax.plot(d["epoch"], d["val_loss"], linestyle="--", label=f"{model} val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{family}: training and validation loss")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / f"{family.lower()}_training_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

def make_pivot(metrics_df, family, metric):
    d = metrics_df[metrics_df["family"] == family]
    if d.empty or metric not in d.columns:
        return None
    piv = d.pivot(index="model", columns="parameter", values=metric)
    return piv.reindex(MODEL_FAMILIES[family])

def plot_metric_heatmap(metrics_df, family, metric, out_dir):
    piv = make_pivot(metrics_df, family, metric)
    if piv is None or piv.empty:
        return
    vals = piv.to_numpy(dtype=float)
    if not np.isfinite(vals).any():
        return
    fig, ax = plt.subplots(figsize=(max(6, 0.9*len(piv.columns)), max(4, 0.8*len(piv.index))))
    im = ax.imshow(vals, aspect="auto")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels([nice_model_name(x) for x in piv.index])
    ax.set_title(f"{family}: {metric}")
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            if np.isfinite(vals[i, j]):
                ax.text(j, i, f"{vals[i,j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label=metric)
    fig.tight_layout()
    fig.savefig(out_dir / f"{family.lower()}_{metric}_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

def summarize_by_model(metrics_df, family):
    d = metrics_df[metrics_df["family"] == family]
    if d.empty:
        return pd.DataFrame()
    cols = discover_metric_columns(d)
    if not cols:
        return pd.DataFrame()
    out = d.groupby("model")[cols].mean(numeric_only=True).reset_index()
    return out.set_index("model").reindex(MODEL_FAMILIES[family]).reset_index()

def plot_model_summary_bars(summary_df, family, out_dir):
    if summary_df.empty:
        return
    metrics = [m for m in ["pearson_r", "rmse", "coverage_90", "posterior_contraction"] if m in summary_df.columns]
    if not metrics:
        return
    fig, axes = plt.subplots(len(metrics), 1, figsize=(8, 3.5*len(metrics)))
    if len(metrics) == 1:
        axes = [axes]
    x = np.arange(len(summary_df))
    labels = [nice_model_name(m) for m in summary_df["model"]]
    for ax, metric in zip(axes, metrics):
        ax.bar(x, summary_df[metric].to_numpy(dtype=float))
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel(metric)
        ax.set_title(f"{family}: mean {metric} across parameters")
        if metric == "coverage_90":
            ax.axhline(0.90, linestyle="--", linewidth=1)
    fig.tight_layout()
    fig.savefig(out_dir / f"{family.lower()}_mean_metric_bars.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

def plot_recovery_scatter_per_model(long_df, family, out_dir):
    d_family = long_df[long_df["family"] == family]
    if d_family.empty:
        return
    for model in MODEL_FAMILIES[family]:
        d = d_family[d_family["model"] == model]
        if d.empty:
            continue
        params = list(pd.unique(d["parameter"]))
        ncols = 3
        nrows = math.ceil(len(params)/ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.3*ncols, 4.0*nrows))
        axes = np.asarray(axes).reshape(-1)
        for ax, param in zip(axes, params):
            sub = d[d["parameter"] == param]
            x = sub["true"].to_numpy(dtype=float)
            y = sub["posterior_median"].to_numpy(dtype=float)
            ax.scatter(x, y, alpha=0.35, s=12)
            lo = np.nanmin([np.nanmin(x), np.nanmin(y)])
            hi = np.nanmax([np.nanmax(x), np.nanmax(y)])
            ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
            title = param
            if len(x) > 1 and np.std(x) > 0 and np.std(y) > 0:
                title += f" (r={np.corrcoef(x, y)[0,1]:.2f})"
            ax.set_xlabel("True")
            ax.set_ylabel("Estimate")
            ax.set_title(title)
        for ax in axes[len(params):]:
            ax.axis("off")
        fig.suptitle(f"{family}: parameter recovery — {model}", y=1.01)
        fig.tight_layout()
        fig.savefig(out_dir / f"{model}_parameter_recovery_scatter.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

def plot_abs_error_boxplots(long_df, family, out_dir):
    d = long_df[long_df["family"] == family].copy()
    if d.empty:
        return
    d["abs_error"] = np.abs(d["posterior_median"] - d["true"])
    params = list(pd.unique(d["parameter"]))
    fig, axes = plt.subplots(len(params), 1, figsize=(8.5, 2.8*len(params)))
    if len(params) == 1:
        axes = [axes]
    for ax, param in zip(axes, params):
        data, labels = [], []
        for model in MODEL_FAMILIES[family]:
            sub = d[(d["model"] == model) & (d["parameter"] == param)]["abs_error"].dropna().to_numpy()
            if len(sub) == 0:
                continue
            data.append(sub)
            labels.append(model.replace("_", "\n"))
        if data:
            ax.boxplot(data, tick_labels=labels, showfliers=False)
        ax.set_ylabel("|error|")
        ax.set_title(f"{family}: absolute error for {param}")
    fig.tight_layout()
    fig.savefig(out_dir / f"{family.lower()}_absolute_error_boxplots.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

def plot_coverage_bars(metrics_df, family, out_dir):
    d = metrics_df[metrics_df["family"] == family]
    if d.empty or "coverage_90" not in d.columns:
        return
    params = list(pd.unique(d["parameter"]))
    width = 0.18
    x = np.arange(len(params))
    fig, ax = plt.subplots(figsize=(max(7.5, 1.0*len(params)+3), 5))
    for i, model in enumerate(MODEL_FAMILIES[family]):
        sub = d[d["model"] == model].set_index("parameter").reindex(params)
        vals = sub["coverage_90"].to_numpy(dtype=float)
        ax.bar(x + (i - 1.5)*width, vals, width=width, label=model)
    ax.axhline(0.90, linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(params, rotation=45, ha="right")
    ax.set_ylabel("Coverage")
    ax.set_title(f"{family}: 90% interval coverage")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"{family.lower()}_coverage90_bars.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

def _prepare_corr_matrix(df):
    if df is None or df.empty:
        return None, None, None
    if df.columns[0] not in df.columns[1:]:
        df2 = df.set_index(df.columns[0])
    else:
        df2 = df.copy()
    return df2.to_numpy(dtype=float), list(df2.index), list(df2.columns)

def plot_posterior_corr_panels(model_outputs, family, key, out_dir):
    mats, labels = [], []
    for d in model_outputs:
        if d["family"] != family:
            continue
        arr, idx, cols = _prepare_corr_matrix(d[key])
        if arr is None:
            continue
        mats.append((arr, idx, cols))
        labels.append(d["model"])
    if not mats:
        return
    n = len(mats)
    ncols = 2
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows))
    axes = np.asarray(axes).reshape(-1)
    if key == "corr_signed":
        vmin, vmax, cbar_label = -1, 1, "Mean signed posterior correlation"
    else:
        vmin, vmax, cbar_label = 0, 1, "Mean absolute posterior correlation"
    first_im = None
    for ax, (arr, idx, cols), label in zip(axes, mats, labels):
        im = ax.imshow(arr, vmin=vmin, vmax=vmax)
        if first_im is None:
            first_im = im
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=45, ha="right")
        ax.set_yticks(range(len(idx)))
        ax.set_yticklabels(idx)
        ax.set_title(label)
    for ax in axes[len(mats):]:
        ax.axis("off")
    fig.colorbar(first_im, ax=axes[:len(mats)], fraction=0.02, pad=0.02, label=cbar_label)
    fig.suptitle(f"{family}: posterior correlation panels ({key})", y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / f"{family.lower()}_{key}_panel.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Additional diagnostics for understanding WHY parameter recovery can be poor
# =============================================================================

def _parameter_axes(params, width=4.3, height=3.8):
    ncols = 3
    nrows = math.ceil(len(params) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(width * ncols, height * nrows))
    return fig, np.asarray(axes).reshape(-1)


def _quantile_bin_summaries(x, y, n_bins=8):
    """Return robust equal-count bin centers, means, and standard errors."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) == 0:
        return np.array([]), np.array([]), np.array([])

    order = np.argsort(x)
    chunks = np.array_split(order, min(n_bins, len(order)))
    centers, means, ses = [], [], []
    for idx in chunks:
        if len(idx) == 0:
            continue
        yy = y[idx]
        centers.append(float(np.mean(x[idx])))
        means.append(float(np.mean(yy)))
        ses.append(float(np.std(yy, ddof=1) / np.sqrt(len(yy))) if len(yy) > 1 else 0.0)
    return np.asarray(centers), np.asarray(means), np.asarray(ses)


def plot_sbc_calibration_ecdf(long_df, family, out_dir):
    """
    SBC rank ECDF difference plots using the fractional posterior ranks saved by s2.

    s2's rank_u = fraction(posterior_draw <= ground_truth), which is exactly the
    fractional rank quantity underlying BayesFlow calibration_ecdf. Using the
    saved ranks avoids re-running posterior sampling. The band shown here is a
    simultaneous 95% Dvoretzky-Kiefer-Wolfowitz band around the ideal uniform ECDF.
    """
    d_family = long_df[long_df["family"] == family]
    if d_family.empty or "rank_u" not in d_family.columns:
        return

    for model in MODEL_FAMILIES[family]:
        d = d_family[d_family["model"] == model]
        if d.empty:
            continue
        params = list(pd.unique(d["parameter"]))
        fig, axes = _parameter_axes(params)

        for ax, param in zip(axes, params):
            u = d.loc[d["parameter"] == param, "rank_u"].to_numpy(dtype=float)
            u = np.sort(np.clip(u[np.isfinite(u)], 0.0, 1.0))
            n = len(u)
            if n == 0:
                ax.axis("off")
                continue

            ecdf = np.arange(1, n + 1, dtype=float) / n
            diff = ecdf - u
            eps = np.sqrt(np.log(2.0 / 0.05) / (2.0 * n))

            ax.fill_between([0.0, 1.0], [-eps, -eps], [eps, eps], alpha=0.20)
            ax.plot(u, diff, linewidth=1.5)
            ax.axhline(0.0, linestyle="--", linewidth=1)
            ax.set_xlim(0, 1)
            ax.set_xlabel("Fractional posterior rank")
            ax.set_ylabel("ECDF - Uniform")
            ax.set_title(f"{param} (n={n})")

        for ax in axes[len(params):]:
            ax.axis("off")
        fig.suptitle(f"{family}: SBC calibration ECDF difference — {model}", y=1.01)
        fig.tight_layout()
        fig.savefig(out_dir / f"{model}_calibration_ecdf.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def plot_sbc_rank_histograms(long_df, family, out_dir, num_bins=10):
    """
    Rank histograms complement ECDF calibration plots.

    Rough interpretation:
      U-shaped     -> posterior often too narrow / overconfident
      mound-shaped -> posterior often too wide / underconfident
      skewed       -> systematic bias
    """
    d_family = long_df[long_df["family"] == family]
    if d_family.empty or "rank_u" not in d_family.columns:
        return

    for model in MODEL_FAMILIES[family]:
        d = d_family[d_family["model"] == model]
        if d.empty:
            continue
        params = list(pd.unique(d["parameter"]))
        fig, axes = _parameter_axes(params)

        for ax, param in zip(axes, params):
            u = d.loc[d["parameter"] == param, "rank_u"].to_numpy(dtype=float)
            u = np.clip(u[np.isfinite(u)], 0.0, 1.0)
            n = len(u)
            if n == 0:
                ax.axis("off")
                continue

            counts, edges = np.histogram(u, bins=num_bins, range=(0.0, 1.0))
            props = counts / n
            centers = (edges[:-1] + edges[1:]) / 2
            width = edges[1] - edges[0]

            expected = 1.0 / num_bins
            # Approximate 99% pointwise binomial band; useful as a visual guide.
            se = np.sqrt(expected * (1.0 - expected) / n)
            lo = max(0.0, expected - 2.576 * se)
            hi = min(1.0, expected + 2.576 * se)

            ax.bar(centers, props, width=0.9 * width)
            ax.axhline(expected, linestyle="--", linewidth=1)
            ax.fill_between([0.0, 1.0], [lo, lo], [hi, hi], alpha=0.16)
            ax.set_xlim(0, 1)
            ax.set_xlabel("Fractional posterior rank")
            ax.set_ylabel("Proportion")
            ax.set_title(param)

        for ax in axes[len(params):]:
            ax.axis("off")
        fig.suptitle(f"{family}: SBC rank histograms — {model}", y=1.01)
        fig.tight_layout()
        fig.savefig(out_dir / f"{model}_calibration_rank_histogram.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def plot_residual_vs_true(long_df, family, out_dir):
    """Diagnose shrinkage, boundary effects, and nonlinear recovery bias."""
    d_family = long_df[long_df["family"] == family]
    if d_family.empty:
        return

    for model in MODEL_FAMILIES[family]:
        d = d_family[d_family["model"] == model].copy()
        if d.empty:
            continue
        d["residual"] = d["posterior_median"] - d["true"]
        params = list(pd.unique(d["parameter"]))
        fig, axes = _parameter_axes(params)

        for ax, param in zip(axes, params):
            sub = d[d["parameter"] == param]
            x = sub["true"].to_numpy(dtype=float)
            r = sub["residual"].to_numpy(dtype=float)
            ax.scatter(x, r, alpha=0.25, s=10)
            ax.axhline(0.0, linestyle="--", linewidth=1)

            bx, by, bse = _quantile_bin_summaries(x, r, n_bins=8)
            if len(bx):
                ax.plot(bx, by, marker="o", linewidth=1.5)
                ax.fill_between(bx, by - bse, by + bse, alpha=0.18)

            ax.set_xlabel(f"True {param}")
            ax.set_ylabel("Posterior median - true")
            ax.set_title(param)

        for ax in axes[len(params):]:
            ax.axis("off")
        fig.suptitle(f"{family}: residual vs true value — {model}", y=1.01)
        fig.tight_layout()
        fig.savefig(out_dir / f"{model}_residual_vs_true.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def plot_conditional_coverage(long_df, family, out_dir):
    """
    Plot empirical 90% coverage over the true-parameter range.
    This catches local failures hidden by acceptable overall coverage.
    """
    d_family = long_df[long_df["family"] == family]
    if d_family.empty or "covered" not in d_family.columns:
        return

    for model in MODEL_FAMILIES[family]:
        d = d_family[d_family["model"] == model]
        if d.empty:
            continue
        params = list(pd.unique(d["parameter"]))
        fig, axes = _parameter_axes(params)

        for ax, param in zip(axes, params):
            sub = d[d["parameter"] == param]
            x = sub["true"].to_numpy(dtype=float)
            covered = sub["covered"].astype(float).to_numpy()
            bx, by, bse = _quantile_bin_summaries(x, covered, n_bins=8)

            if len(bx):
                ax.plot(bx, by, marker="o", linewidth=1.5)
                ax.fill_between(bx, np.maximum(0, by - bse), np.minimum(1, by + bse), alpha=0.18)
            ax.axhline(0.90, linestyle="--", linewidth=1)
            ax.set_ylim(0, 1)
            ax.set_xlabel(f"True {param}")
            ax.set_ylabel("Empirical 90% coverage")
            ax.set_title(param)

        for ax in axes[len(params):]:
            ax.axis("off")
        fig.suptitle(f"{family}: conditional interval coverage — {model}", y=1.01)
        fig.tight_layout()
        fig.savefig(out_dir / f"{model}_conditional_coverage.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def plot_uncertainty_vs_error(long_df, family, out_dir):
    """
    Compare posterior uncertainty with realized recovery error.
    If large errors systematically occur with tiny posterior SD, the amortizer is overconfident.
    """
    d_family = long_df[long_df["family"] == family]
    if d_family.empty or "posterior_sd" not in d_family.columns:
        return

    for model in MODEL_FAMILIES[family]:
        d = d_family[d_family["model"] == model]
        if d.empty:
            continue
        params = list(pd.unique(d["parameter"]))
        fig, axes = _parameter_axes(params)

        for ax, param in zip(axes, params):
            sub = d[d["parameter"] == param]
            sd = sub["posterior_sd"].to_numpy(dtype=float)
            err = np.abs(
                sub["posterior_median"].to_numpy(dtype=float)
                - sub["true"].to_numpy(dtype=float)
            )
            ok = np.isfinite(sd) & np.isfinite(err)
            sd, err = sd[ok], err[ok]
            if len(sd):
                ax.scatter(sd, err, alpha=0.25, s=10)
                hi = max(float(np.max(sd)), float(np.max(err)))
                ax.plot([0, hi], [0, hi], linestyle="--", linewidth=1)
            ax.set_xlabel("Posterior SD")
            ax.set_ylabel("|posterior median - true|")
            ax.set_title(param)

        for ax in axes[len(params):]:
            ax.axis("off")
        fig.suptitle(f"{family}: posterior uncertainty vs realized error — {model}", y=1.01)
        fig.tight_layout()
        fig.savefig(out_dir / f"{model}_uncertainty_vs_error.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def plot_standardized_error(long_df, family, out_dir):
    """
    Heuristic standardized recovery error:
        (posterior median - true) / posterior SD
    Large shifts from zero indicate bias; very large spread flags overconfidence.
    """
    d_family = long_df[long_df["family"] == family]
    if d_family.empty or "posterior_sd" not in d_family.columns:
        return

    for model in MODEL_FAMILIES[family]:
        d = d_family[d_family["model"] == model].copy()
        if d.empty:
            continue
        sd = d["posterior_sd"].to_numpy(dtype=float)
        err = (
            d["posterior_median"].to_numpy(dtype=float)
            - d["true"].to_numpy(dtype=float)
        )
        d["standardized_error"] = np.divide(
            err,
            sd,
            out=np.full_like(err, np.nan, dtype=float),
            where=np.isfinite(sd) & (sd > 0),
        )

        params = list(pd.unique(d["parameter"]))
        arrays, labels = [], []
        for p in params:
            z = d.loc[d["parameter"] == p, "standardized_error"].to_numpy(dtype=float)
            z = z[np.isfinite(z)]
            if len(z):
                arrays.append(z)
                labels.append(p)

        if not arrays:
            continue
        fig, ax = plt.subplots(figsize=(max(7.0, 1.0 * len(labels)), 5.0))
        ax.boxplot(arrays, tick_labels=labels, showfliers=False)
        ax.axhline(0.0, linestyle="--", linewidth=1)
        ax.axhline(1.0, linestyle=":", linewidth=1)
        ax.axhline(-1.0, linestyle=":", linewidth=1)
        ax.tick_params(axis="x", rotation=45)
        ax.set_ylabel("(posterior median - true) / posterior SD")
        ax.set_title(f"{family}: standardized recovery error — {model}")
        fig.tight_layout()
        fig.savefig(out_dir / f"{model}_standardized_error.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def plot_recovery_error_correlation(long_df, family, out_dir):
    """
    Correlation of recovery residuals across parameters.
    Strong off-diagonal structure is a direct clue for parameter trade-offs /
    weak identifiability across simulated data sets.
    """
    d_family = long_df[long_df["family"] == family]
    if d_family.empty:
        return

    for model in MODEL_FAMILIES[family]:
        d = d_family[d_family["model"] == model].copy()
        if d.empty:
            continue
        d["residual"] = d["posterior_median"] - d["true"]
        piv = d.pivot(index="participant_id", columns="parameter", values="residual")
        corr = piv.corr()
        vals = corr.to_numpy(dtype=float)

        fig, ax = plt.subplots(figsize=(max(6, 0.8 * len(corr.columns)), max(5, 0.7 * len(corr.columns))))
        im = ax.imshow(vals, vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(corr.index)))
        ax.set_yticklabels(corr.index)
        ax.set_title(f"Recovery residual correlation — {model}")
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                if np.isfinite(vals[i, j]):
                    ax.text(j, i, f"{vals[i, j]:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, label="Residual correlation")
        fig.tight_layout()
        fig.savefig(out_dir / f"{model}_recovery_error_correlation.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def compute_extra_diagnostic_metrics(long_df):
    rows = []
    if long_df.empty:
        return pd.DataFrame()

    for (family, model, parameter), d in long_df.groupby(["family", "model", "parameter"]):
        true = d["true"].to_numpy(dtype=float)
        est = d["posterior_median"].to_numpy(dtype=float)
        resid = est - true
        post_sd = d["posterior_sd"].to_numpy(dtype=float) if "posterior_sd" in d else np.full(len(d), np.nan)

        if len(true) > 2 and np.std(true) > 0:
            slope = float(np.cov(true, est, ddof=1)[0, 1] / np.var(true, ddof=1))
            intercept = float(np.mean(est) - slope * np.mean(true))
            residual_true_corr = float(np.corrcoef(true, resid)[0, 1]) if np.std(resid) > 0 else np.nan
        else:
            slope, intercept, residual_true_corr = np.nan, np.nan, np.nan

        z = np.divide(
            resid,
            post_sd,
            out=np.full_like(resid, np.nan, dtype=float),
            where=np.isfinite(post_sd) & (post_sd > 0),
        )
        abs_err = np.abs(resid)
        if np.sum(np.isfinite(post_sd) & np.isfinite(abs_err)) > 2:
            ok = np.isfinite(post_sd) & np.isfinite(abs_err)
            uncertainty_error_corr = (
                float(np.corrcoef(post_sd[ok], abs_err[ok])[0, 1])
                if np.std(post_sd[ok]) > 0 and np.std(abs_err[ok]) > 0
                else np.nan
            )
        else:
            uncertainty_error_corr = np.nan

        coverage_range = np.nan
        if "covered" in d.columns and len(d) >= 8:
            bx, by, _ = _quantile_bin_summaries(true, d["covered"].astype(float).to_numpy(), n_bins=8)
            if len(by):
                coverage_range = float(np.nanmax(by) - np.nanmin(by))

        rows.append(
            dict(
                family=family,
                model=model,
                parameter=parameter,
                recovery_slope=slope,
                recovery_intercept=intercept,
                residual_true_corr=residual_true_corr,
                mean_standardized_error=float(np.nanmean(z)) if np.isfinite(z).any() else np.nan,
                sd_standardized_error=float(np.nanstd(z, ddof=1)) if np.isfinite(z).sum() > 1 else np.nan,
                uncertainty_error_corr=uncertainty_error_corr,
                conditional_coverage_range=coverage_range,
            )
        )

    return pd.DataFrame(rows)


def plot_extra_metric_heatmaps(extra_df, family, out_dir):
    if extra_df.empty:
        return
    metrics = [
        "recovery_slope",
        "residual_true_corr",
        "mean_standardized_error",
        "sd_standardized_error",
        "conditional_coverage_range",
    ]
    d_family = extra_df[extra_df["family"] == family]
    if d_family.empty:
        return

    for metric in metrics:
        piv = d_family.pivot(index="model", columns="parameter", values=metric)
        piv = piv.reindex(MODEL_FAMILIES[family])
        vals = piv.to_numpy(dtype=float)
        if not np.isfinite(vals).any():
            continue

        fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(piv.columns)), max(4, 0.8 * len(piv.index))))
        im = ax.imshow(vals, aspect="auto")
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels(piv.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels([nice_model_name(x) for x in piv.index])
        ax.set_title(f"{family}: {metric}")
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                if np.isfinite(vals[i, j]):
                    ax.text(j, i, f"{vals[i, j]:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, label=metric)
        fig.tight_layout()
        fig.savefig(out_dir / f"{family.lower()}_{metric}_diagnostic_heatmap.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def write_summary_json(metrics_df, out_dir):
    summary = {}
    if not metrics_df.empty:
        for family, models in MODEL_FAMILIES.items():
            d = metrics_df[metrics_df["family"] == family]
            if d.empty:
                continue
            cols = discover_metric_columns(d)
            fam = {}
            for metric in cols:
                by_model = d.groupby("model")[metric].mean(numeric_only=True).reindex(models)
                if by_model.dropna().empty:
                    best = None
                else:
                    best = by_model.idxmin() if metric in LOWER_IS_BETTER else by_model.idxmax()
                fam[metric] = {
                    "best_model": best,
                    "values": {k: (None if pd.isna(v) else float(v)) for k, v in by_model.items()}
                }
            summary[family] = fam
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

def make_family_package(metrics_df, long_df, history_df, model_outputs, family, out_root):
    out_dir = ensure_dir(out_root / family.lower())
    plot_training_curves(history_df, family, out_dir)
    for metric in HEATMAP_METRICS:
        if metric in metrics_df.columns:
            plot_metric_heatmap(metrics_df, family, metric, out_dir)
    summary_df = summarize_by_model(metrics_df, family)
    if not summary_df.empty:
        summary_df.to_csv(out_dir / f"{family.lower()}_model_summary.csv", index=False)
    plot_model_summary_bars(summary_df, family, out_dir)
    plot_recovery_scatter_per_model(long_df, family, out_dir)
    plot_abs_error_boxplots(long_df, family, out_dir)
    plot_coverage_bars(metrics_df, family, out_dir)
    plot_posterior_corr_panels(model_outputs, family, "corr_signed", out_dir)
    plot_posterior_corr_panels(model_outputs, family, "corr_abs", out_dir)
    # Additional diagnostics; all existing plots above are intentionally unchanged.
    plot_sbc_calibration_ecdf(long_df, family, out_dir)
    plot_sbc_rank_histograms(long_df, family, out_dir)
    plot_residual_vs_true(long_df, family, out_dir)
    plot_conditional_coverage(long_df, family, out_dir)
    plot_uncertainty_vs_error(long_df, family, out_dir)
    plot_standardized_error(long_df, family, out_dir)
    plot_recovery_error_correlation(long_df, family, out_dir)

def make_overall_family_summary(metrics_df, out_dir):
    if metrics_df.empty or "pearson_r" not in metrics_df.columns or "rmse" not in metrics_df.columns:
        return
    fams, mean_r, mean_rmse = [], [], []
    for family in ["DFT", "RUDT"]:
        d = metrics_df[metrics_df["family"] == family]
        if d.empty:
            continue
        fams.append(family)
        mean_r.append(float(d["pearson_r"].mean()))
        mean_rmse.append(float(d["rmse"].mean()))
    x = np.arange(len(fams))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.bar(x - width/2, mean_r, width=width, label="mean Pearson r")
    ax.bar(x + width/2, mean_rmse, width=width, label="mean RMSE")
    ax.set_xticks(x)
    ax.set_xticklabels(fams)
    ax.set_title("Overall family-level recovery summary")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "overall_family_recovery_summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(
        description="Run/collect parameter recovery for all 8 trained models and create DFT/RUDT visual summaries."
    )
    parser.add_argument("--mode", default="formal", choices=["test", "formal"])
    parser.add_argument("--output-root", default="output")
    parser.add_argument("--n-test", type=int, default=None)
    parser.add_argument("--posterior-draws", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--force-recovery",
        action="store_true",
        help="Re-run parameter recovery even when diagnostics already exist.",
    )
    parser.add_argument(
        "--skip-recovery",
        action="store_true",
        help="Do not run s2 automatically; only summarize existing diagnostics.",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root)
    summary_root = ensure_dir(output_root / "all_models_visualization" / args.mode)

    if args.mode == "formal":
        n_test = 1000 if args.n_test is None else args.n_test
        posterior_draws = 1000 if args.posterior_draws is None else args.posterior_draws
        batch_size = 25 if args.batch_size is None else args.batch_size
    else:
        n_test = 32 if args.n_test is None else args.n_test
        posterior_draws = 100 if args.posterior_draws is None else args.posterior_draws
        batch_size = 16 if args.batch_size is None else args.batch_size

    if not args.skip_recovery:
        run_missing_recoveries(
            output_root=output_root,
            mode=args.mode,
            n_test=n_test,
            posterior_draws=posterior_draws,
            batch_size=batch_size,
            force=args.force_recovery,
        )

    model_outputs = [gather_model_outputs(output_root, model, args.mode) for model in ALL_MODELS]
    metrics_df = concat_metrics(model_outputs)
    long_df = concat_long(model_outputs)
    history_df = concat_history(model_outputs)

    if metrics_df.empty:
        raise RuntimeError(
            "No parameter-recovery metrics were found. "
            "Run without --skip-recovery, or run s2_parameter_recovery.py first."
        )
    if long_df.empty:
        raise RuntimeError(
            "No parameter_recovery_long.csv files were found after recovery."
        )

    expected_families = {"DFT", "RUDT"}
    got_families = set(metrics_df["family"].dropna().unique())
    if not expected_families.issubset(got_families):
        raise RuntimeError(
            f"Incomplete recovery results: expected families {expected_families}, got {got_families}."
        )

    save_tables(metrics_df, long_df, history_df, summary_root)
    write_summary_json(metrics_df, summary_root)

    extra_diagnostic_metrics = compute_extra_diagnostic_metrics(long_df)
    if not extra_diagnostic_metrics.empty:
        extra_diagnostic_metrics.to_csv(
            summary_root / "all_models_extra_diagnostic_metrics.csv",
            index=False,
        )

    for family in ["DFT", "RUDT"]:
        make_family_package(metrics_df, long_df, history_df, model_outputs, family, summary_root)
        plot_extra_metric_heatmaps(
            extra_diagnostic_metrics,
            family,
            summary_root / family.lower(),
        )

    make_overall_family_summary(metrics_df, summary_root)

    print("\n" + "=" * 80)
    print("ALL 8 MODELS COMPLETE")
    print("Mode:", args.mode)
    print("Parameter recovery:")
    print(f"  n_test={n_test}, posterior_draws={posterior_draws}, batch_size={batch_size}")
    print("Combined visualization output:")
    print(" ", summary_root)
    print("=" * 80)


if __name__ == "__main__":
    main()
