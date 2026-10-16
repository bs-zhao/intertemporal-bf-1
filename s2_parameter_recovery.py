#!/usr/bin/env python
#SBATCH --account=PDE0114
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --job-name=bf8-recovery
#SBATCH --output=slurm-%j.out

"""Parameter recovery and diagnostics for one trained eight-model posterior."""

import os
os.environ.setdefault("KERAS_BACKEND", "torch")

import argparse
import gc
import json
import math
from pathlib import Path
from importlib.metadata import version as package_version

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, kstest
import bayesflow as bf
import keras
import simulators_8models as sim

BF_REQUIRED_VERSION = "2.0.13"
BASE_SEED = 20261015
TEST_CFG = dict(n_test=32, posterior_draws=100, batch_size=16)
FORMAL_CFG = dict(n_test=1000, posterior_draws=1000, batch_size=25)
CI_LEVEL = 0.90
OUTPUT_ROOT = Path("output")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=sim.MODEL_NAMES)
    p.add_argument("--mode", required=True, choices=("test", "formal"))
    p.add_argument("--n-test", type=int)
    p.add_argument("--posterior-draws", type=int)
    p.add_argument("--batch-size", type=int)
    return p.parse_args()


def check_environment():
    if not Path("simulators_8models.py").exists():
        raise RuntimeError("Run this script from the repository root.")
    v = package_version("bayesflow")
    if v != BF_REQUIRED_VERSION:
        raise RuntimeError(f"Expected BayesFlow {BF_REQUIRED_VERSION}, found {v}.")
    if keras.backend.backend() != "torch":
        raise RuntimeError(f"Expected Keras torch backend, found {keras.backend.backend()}.")


def cleanup():
    gc.collect()
    try:
        import torch
        if torch.cuda.device_count() > 0:
            torch.cuda.empty_cache()
    except Exception:
        pass


def as_draw_matrix(x, n):
    x = np.asarray(x)
    while x.ndim > 2 and x.shape[-1] == 1:
        x = np.squeeze(x, axis=-1)
    if x.ndim == 1:
        x = x[None, :]
    if x.shape[0] != n and x.ndim == 2 and x.shape[1] == n:
        x = x.T
    if x.ndim != 2 or x.shape[0] != n:
        raise ValueError(f"Unexpected posterior shape {x.shape}; expected dataset x draw.")
    return x.astype(float)


def as_truth(x, n):
    x = np.asarray(x).squeeze()
    if x.ndim == 0:
        x = x[None]
    if x.ndim != 1 or len(x) != n:
        raise ValueError(f"Unexpected truth shape {x.shape}.")
    return x.astype(float)


def safe_corr(fun, x, y):
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    try:
        return float(fun(x, y)[0])
    except Exception:
        return np.nan


def collect(approximator, simulator, params, cfg):
    rows = []
    corr_sum = np.zeros((len(params), len(params)))
    abs_sum = np.zeros_like(corr_sum)
    corr_n = np.zeros_like(corr_sum)
    done = 0

    while done < cfg["n_test"]:
        b = min(cfg["batch_size"], cfg["n_test"] - done)
        raw = simulator.sample(b)
        samples = approximator.sample(
            conditions={"trials": np.asarray(raw["trials"], dtype="float32")},
            num_samples=cfg["posterior_draws"],
        )
        truths = {p: as_truth(raw[p], b) for p in params}
        mats = [as_draw_matrix(samples[p], b) for p in params]
        post = np.stack(mats, axis=-1)  # batch x draw x parameter

        for i in range(b):
            c = np.corrcoef(post[i], rowvar=False)
            finite = np.isfinite(c)
            corr_sum[finite] += c[finite]
            abs_sum[finite] += np.abs(c[finite])
            corr_n[finite] += 1

            for j, p in enumerate(params):
                d = post[i, :, j]
                true = float(truths[p][i])
                qlo, qhi = np.quantile(d, [(1-CI_LEVEL)/2, 1-(1-CI_LEVEL)/2])
                med = float(np.median(d))
                row = dict(
                    participant_id=done+i,
                    parameter=p,
                    true=true,
                    posterior_median=med,
                    posterior_mean=float(np.mean(d)),
                    posterior_sd=float(np.std(d, ddof=1)),
                    ci_lower=float(qlo),
                    ci_upper=float(qhi),
                    covered=bool(qlo <= true <= qhi),
                    rank_u=float(np.mean(d <= true)),
                )
                if p == "n":
                    rounded = float(np.clip(np.rint(med), *sim.RUDT_N_RANGE))
                    row["posterior_median_rounded"] = rounded
                    row["n_exact_rounded"] = bool(rounded == true)
                rows.append(row)

        done += b
        print(f"recovery {done}/{cfg['n_test']}")
        del raw, samples, truths, mats, post
        cleanup()

    mean_corr = np.divide(corr_sum, corr_n, out=np.full_like(corr_sum, np.nan), where=corr_n>0)
    mean_abs = np.divide(abs_sum, corr_n, out=np.full_like(abs_sum, np.nan), where=corr_n>0)
    return pd.DataFrame(rows), mean_corr, mean_abs


def metrics_table(df, params):
    out = []
    for p in params:
        d = df[df.parameter == p]
        true = d.true.to_numpy(float)
        est = d.posterior_median.to_numpy(float)
        err = est - true
        rmse = float(np.sqrt(np.mean(err**2)))
        true_sd = float(np.std(true, ddof=1)) if len(true) > 1 else np.nan
        prior_var = float(np.var(true, ddof=1)) if len(true) > 1 else np.nan
        post_var = float(np.mean(d.posterior_sd.to_numpy(float)**2))
        try:
            ks = kstest(d.rank_u.to_numpy(float), "uniform")
            ks_stat, ks_p = float(ks.statistic), float(ks.pvalue)
        except Exception:
            ks_stat, ks_p = np.nan, np.nan
        row = dict(
            parameter=p,
            n=len(d),
            pearson_r=safe_corr(pearsonr, true, est),
            spearman_rho=safe_corr(spearmanr, true, est),
            rmse=rmse,
            nrmse_by_true_sd=rmse/true_sd if np.isfinite(true_sd) and true_sd>0 else np.nan,
            mae=float(np.mean(np.abs(err))),
            bias=float(np.mean(err)),
            coverage_90=float(np.mean(d.covered.astype(float))),
            mean_ci_width_90=float(np.mean(d.ci_upper-d.ci_lower)),
            posterior_contraction=(1-post_var/prior_var) if np.isfinite(prior_var) and prior_var>0 else np.nan,
            rank_uniform_ks=ks_stat,
            rank_uniform_p=ks_p,
        )
        if p == "n":
            row["rounded_exact_accuracy"] = float(np.mean(d.n_exact_rounded.astype(float)))
            rounded = d.posterior_median_rounded.to_numpy(float)
            row["rounded_rmse"] = float(np.sqrt(np.mean((rounded-true)**2)))
        out.append(row)
    return pd.DataFrame(out)


def save_heatmap(matrix, names, title, path, signed):
    fig, ax = plt.subplots(figsize=(max(6, .8*len(names)), max(5, .7*len(names))))
    im = ax.imshow(matrix, vmin=-1 if signed else 0, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_title(title)
    for i in range(len(names)):
        for j in range(len(names)):
            if np.isfinite(matrix[i,j]):
                ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax)
    fig.tight_layout(); fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def make_plots(df, metrics, params, model, fig_dir, mean_corr, mean_abs):
    ncols, nrows = 3, math.ceil(len(params)/3)
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.2*nrows)); axes=np.asarray(axes).reshape(-1)
    for ax, p in zip(axes, params):
        d=df[df.parameter==p]; x=d.true.to_numpy(float); y=d.posterior_median.to_numpy(float)
        ax.scatter(x,y,s=16,alpha=.55); lo=min(x.min(),y.min()); hi=max(x.max(),y.max()); ax.plot([lo,hi],[lo,hi],"--",lw=1)
        r=safe_corr(pearsonr,x,y); ax.set_xlabel(f"True {p}"); ax.set_ylabel(f"Posterior median {p}"); ax.set_title(f"{p}: r={r:.3f}" if np.isfinite(r) else p)
    for ax in axes[len(params):]: ax.axis("off")
    fig.suptitle(f"Parameter recovery — {model}"); fig.tight_layout(); fig.savefig(fig_dir/"parameter_recovery_scatter.png",dpi=180,bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(max(7,.9*len(params)),4.5)); ax.bar(metrics.parameter, metrics.coverage_90); ax.axhline(CI_LEVEL,ls="--"); ax.set_ylim(0,1); ax.tick_params(axis="x",rotation=45); ax.set_title("90% posterior interval coverage"); fig.tight_layout(); fig.savefig(fig_dir/"coverage_90.png",dpi=180,bbox_inches="tight"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(max(7,.9*len(params)),4.5)); ax.bar(metrics.parameter, metrics.posterior_contraction); ax.axhline(0,ls="--"); ax.tick_params(axis="x",rotation=45); ax.set_title("Posterior contraction"); fig.tight_layout(); fig.savefig(fig_dir/"posterior_contraction.png",dpi=180,bbox_inches="tight"); plt.close(fig)

    save_heatmap(mean_corr, params, "Mean signed within-dataset posterior correlation", fig_dir/"posterior_correlation_mean_signed.png", True)
    save_heatmap(mean_abs, params, "Mean absolute within-dataset posterior correlation", fig_dir/"posterior_correlation_mean_abs.png", False)

    if "n" in params:
        d=df[df.parameter=="n"]; vals=np.arange(sim.RUDT_N_RANGE[0],sim.RUDT_N_RANGE[1]+1); mat=np.zeros((len(vals),len(vals)))
        for t,p in zip(d.true.astype(int), d.posterior_median_rounded.astype(int)):
            if t in vals and p in vals: mat[t-vals[0],p-vals[0]] += 1
        denom=mat.sum(1,keepdims=True); prop=np.divide(mat,denom,out=np.zeros_like(mat),where=denom>0)
        fig,ax=plt.subplots(figsize=(7,6)); im=ax.imshow(prop,vmin=0,vmax=1); ax.set_xticks(range(len(vals)),vals); ax.set_yticks(range(len(vals)),vals); ax.set_xlabel("Rounded posterior median n"); ax.set_ylabel("True n"); ax.set_title("Discrete n recovery"); fig.colorbar(im,ax=ax); fig.tight_layout(); fig.savefig(fig_dir/"n_recovery_confusion.png",dpi=180,bbox_inches="tight"); plt.close(fig)


def plot_training_history(run_dir, fig_dir):
    p=run_dir/"training_history.csv"
    if not p.exists(): return
    h=pd.read_csv(p); fig,ax=plt.subplots(figsize=(7,4.5))
    if "loss" in h: ax.plot(np.arange(1,len(h)+1),h.loss,label="train")
    if "val_loss" in h: ax.plot(np.arange(1,len(h)+1),h.val_loss,label="validation")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.legend(); fig.tight_layout(); fig.savefig(fig_dir/"training_loss.png",dpi=180,bbox_inches="tight"); plt.close(fig)
    if "learning_rate" in h:
        fig,ax=plt.subplots(figsize=(7,4)); ax.plot(np.arange(1,len(h)+1),h.learning_rate); ax.set_xlabel("Epoch"); ax.set_ylabel("Learning rate"); fig.tight_layout(); fig.savefig(fig_dir/"learning_rate.png",dpi=180,bbox_inches="tight"); plt.close(fig)


def main():
    args=parse_args(); check_environment(); params=sim.PARAM_NAMES[args.model]
    cfg=(TEST_CFG if args.mode=="test" else FORMAL_CFG).copy()
    if args.n_test is not None: cfg["n_test"]=args.n_test
    if args.posterior_draws is not None: cfg["posterior_draws"]=args.posterior_draws
    if args.batch_size is not None: cfg["batch_size"]=args.batch_size

    run_dir=OUTPUT_ROOT/args.model/args.mode; model_path=run_dir/"posterior_final.keras"
    if not model_path.exists(): raise FileNotFoundError(f"Missing {model_path}; run s1 first.")
    diag=run_dir/"diagnostics"; figs=diag/"figures"; figs.mkdir(parents=True,exist_ok=True)
    approximator=keras.saving.load_model(model_path)
    idx=sim.MODEL_NAMES.index(args.model)
    simulator, redraw=sim.make_bayesflow_parameter_simulator(bf,args.model,BASE_SEED+100000+1000*idx)
    plot_training_history(run_dir,figs)
    df,mean_corr,mean_abs=collect(approximator,simulator,params,cfg); metrics=metrics_table(df,params)
    df.to_csv(diag/"parameter_recovery_long.csv",index=False); metrics.to_csv(diag/"parameter_recovery_metrics.csv",index=False)
    pd.DataFrame(mean_corr,index=params,columns=params).to_csv(diag/"posterior_correlation_mean_signed.csv")
    pd.DataFrame(mean_abs,index=params,columns=params).to_csv(diag/"posterior_correlation_mean_abs.csv")
    make_plots(df,metrics,params,args.model,figs,mean_corr,mean_abs)
    summary=dict(model=args.model,mode=args.mode,parameters=list(params),n_trials=len(sim.DEMURIE_DESIGN),evaluation_config=cfg,trained_model=str(model_path),fresh_test_redraw_stats=redraw,ter_pop_frozen=list(sim.TER_POP))
    (diag/"diagnostics_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(metrics.to_string(index=False)); print("outputs:",diag)
    del approximator; keras.backend.clear_session(); cleanup()


if __name__ == "__main__":
    main()
