#!/usr/bin/env python

"""
Model recovery for the trained 8-class BayesFlow classifier.

This script loads classifier_best.keras produced by
s4_train_model_classifier.py, generates fresh balanced held-out data for each
of the eight models, estimates posterior model probabilities, and saves:

- 8 x 8 confusion matrix (counts and row-normalized)
- 2 x 2 family-level DFT-vs-RUDT confusion matrix
- mean posterior model-probability matrix
- per-model recovery metrics
- per-dataset predictions and model probabilities

Because s4 writes classifier_best.keras during training using an atomic
checkpoint replacement, this script can be run after the first completed best
checkpoint even if s4 has not yet finished training.

Examples
--------
python s5_model_recovery.py --mode test
python s5_model_recovery.py --mode formal
"""

import os
os.environ.setdefault("KERAS_BACKEND", "torch")

import argparse
import gc
import json
from pathlib import Path
from importlib.metadata import version as package_version

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
import bayesflow as bf

import simulators_8models as sim


BF_REQUIRED_VERSION = "2.0.13"
BASE_SEED = 20261017

DEFAULTS = {
    "test": dict(n_per_model=32, batch_size=32),
    "formal": dict(n_per_model=1000, batch_size=100),
}

OUTPUT_ROOT = Path("output") / "model_recovery"

DISPLAY_NAMES = [
    "DFT basic none",
    "DFT basic fixed",
    "DFT basic driftsign",
    "DFT modified none",
    "RUDT sym fixed",
    "RUDT sym decreasing",
    "RUDT asym fixed",
    "RUDT asym decreasing",
]


def check_environment():
    if not Path("simulators_8models.py").exists():
        raise RuntimeError("Run this script from the repository root.")

    v = package_version("bayesflow")
    if v != BF_REQUIRED_VERSION:
        raise RuntimeError(f"Expected BayesFlow {BF_REQUIRED_VERSION}, found {v}.")
    if keras.backend.backend() != "torch":
        raise RuntimeError(
            f"Expected Keras torch backend, found {keras.backend.backend()!r}."
        )

    print("BayesFlow:", v)
    print("Keras backend:", keras.backend.backend())


def cleanup():
    gc.collect()
    try:
        import torch
        if torch.cuda.device_count() > 0:
            torch.cuda.empty_cache()
    except Exception:
        pass


def make_model_only_simulator(model, seed):
    rng = np.random.default_rng(seed)
    redraw_stats = {
        "p_stop": 0,
        "slow": 0,
        "one-sided": 0,
        "accepted": 0,
    }

    def simulate_one():
        _, sim_df = sim.draw_screened_subject(
            model=model,
            rng=rng,
            design_df=sim.DEMURIE_DESIGN,
            redraw_stats=redraw_stats,
        )
        redraw_stats["accepted"] += 1
        return {"trials": sim.encode_trials(sim_df)}

    return bf.make_simulator(simulate_one), redraw_stats


def parse_args():
    p = argparse.ArgumentParser(
        description="Held-out model recovery for the 8-class BayesFlow classifier."
    )
    p.add_argument("--mode", required=True, choices=("test", "formal"))
    p.add_argument("--n-per-model", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument(
        "--classifier",
        type=str,
        default=None,
        help="Optional explicit .keras classifier path. Default: classifier_best.keras.",
    )
    return p.parse_args()


def family_index(model_index):
    return 0 if model_index < 4 else 1


def entropy_rows(probs):
    p = np.clip(np.asarray(probs, dtype=float), 1e-12, 1.0)
    return -np.sum(p * np.log(p), axis=1)


def classify_fresh_data(approximator, n_per_model, batch_size):
    n_models = len(sim.MODEL_NAMES)
    confusion = np.zeros((n_models, n_models), dtype=int)
    family_confusion = np.zeros((2, 2), dtype=int)
    prob_sum = np.zeros((n_models, n_models), dtype=float)
    rows = []
    redraw_all = {}

    global_id = 0

    for true_idx, model in enumerate(sim.MODEL_NAMES):
        simulator, redraw_stats = make_model_only_simulator(
            model=model,
            seed=BASE_SEED + 10000 * (true_idx + 1),
        )
        redraw_all[model] = redraw_stats

        done = 0
        print("=" * 80)
        print(f"TRUE MODEL [{true_idx + 1}/8]: {model}")
        print("=" * 80)

        while done < n_per_model:
            b = min(batch_size, n_per_model - done)
            raw = simulator.sample(b)
            trials = np.asarray(raw["trials"], dtype="float32")

            estimates = approximator.estimate(
                conditions={"trials": trials},
            )
            probs = np.asarray(estimates["model_probs"], dtype=float)

            expected = (b, n_models)
            if probs.shape != expected:
                raise RuntimeError(
                    f"Expected model_probs shape {expected}, got {probs.shape}."
                )

            pred = np.argmax(probs, axis=1)
            max_prob = np.max(probs, axis=1)
            true_prob = probs[:, true_idx]
            ent = entropy_rows(probs)

            for i in range(b):
                pred_idx = int(pred[i])
                confusion[true_idx, pred_idx] += 1
                family_confusion[
                    family_index(true_idx),
                    family_index(pred_idx),
                ] += 1
                prob_sum[true_idx] += probs[i]

                row = {
                    "dataset_id": global_id,
                    "true_index": true_idx,
                    "true_model": model,
                    "pred_index": pred_idx,
                    "pred_model": sim.MODEL_NAMES[pred_idx],
                    "correct": bool(pred_idx == true_idx),
                    "true_family": "DFT" if true_idx < 4 else "RUDT",
                    "pred_family": "DFT" if pred_idx < 4 else "RUDT",
                    "family_correct": bool(family_index(true_idx) == family_index(pred_idx)),
                    "max_probability": float(max_prob[i]),
                    "true_model_probability": float(true_prob[i]),
                    "entropy": float(ent[i]),
                }
                for j, name in enumerate(sim.MODEL_NAMES):
                    row[f"p_{name}"] = float(probs[i, j])
                rows.append(row)
                global_id += 1

            done += b
            print(f"  {done}/{n_per_model}")
            del raw, trials, estimates, probs
            cleanup()

    mean_prob = prob_sum / float(n_per_model)
    return (
        pd.DataFrame(rows),
        confusion,
        family_confusion,
        mean_prob,
        redraw_all,
    )


def row_normalize(matrix):
    matrix = np.asarray(matrix, dtype=float)
    denom = matrix.sum(axis=1, keepdims=True)
    return np.divide(
        matrix,
        denom,
        out=np.zeros_like(matrix, dtype=float),
        where=denom > 0,
    )


def save_matrix_csv(matrix, row_names, col_names, path):
    pd.DataFrame(
        matrix,
        index=row_names,
        columns=col_names,
    ).to_csv(path)


def draw_heatmap(matrix, row_labels, col_labels, title, path, fmt, vmin=None, vmax=None):
    matrix = np.asarray(matrix)
    fig, ax = plt.subplots(
        figsize=(max(9, 1.05 * len(col_labels)), max(7, 0.85 * len(row_labels)))
    )
    im = ax.imshow(matrix, aspect="auto", vmin=vmin, vmax=vmax)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("Predicted model")
    ax.set_ylabel("True model")
    ax.set_title(title)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                format(matrix[i, j], fmt),
                ha="center",
                va="center",
                fontsize=8,
            )

    if matrix.shape == (8, 8):
        ax.axhline(3.5, linewidth=2)
        ax.axvline(3.5, linewidth=2)

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_per_model_metrics(predictions):
    rows = []
    for model in sim.MODEL_NAMES:
        d = predictions[predictions["true_model"] == model]
        rows.append(
            {
                "model": model,
                "family": "DFT" if model.startswith("dft_") else "RUDT",
                "n": len(d),
                "recall_exact_model": float(d["correct"].mean()),
                "recall_family": float(d["family_correct"].mean()),
                "mean_true_model_probability": float(d["true_model_probability"].mean()),
                "mean_max_probability": float(d["max_probability"].mean()),
                "mean_entropy": float(d["entropy"].mean()),
            }
        )
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    check_environment()

    cfg = DEFAULTS[args.mode].copy()
    if args.n_per_model is not None:
        cfg["n_per_model"] = args.n_per_model
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size

    run_dir = OUTPUT_ROOT / args.mode
    classifier_path = (
        Path(args.classifier)
        if args.classifier is not None
        else run_dir / "classifier_best.keras"
    )

    if not classifier_path.exists():
        raise FileNotFoundError(
            f"Missing classifier: {classifier_path}. "
            "Start s4_train_model_classifier.py and wait until at least one "
            "best checkpoint has been written."
        )

    recovery_dir = run_dir / "recovery"
    fig_dir = recovery_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("Loading classifier:", classifier_path)
    approximator = keras.saving.load_model(classifier_path)

    # Smoke check on one fresh dataset from model 0.
    smoke_sim, _ = make_model_only_simulator(
        sim.MODEL_NAMES[0],
        BASE_SEED + 999999,
    )
    smoke = smoke_sim.sample(2)
    smoke_est = approximator.estimate(
        conditions={"trials": np.asarray(smoke["trials"], dtype="float32")}
    )
    smoke_probs = np.asarray(smoke_est["model_probs"])
    expected_smoke = (2, len(sim.MODEL_NAMES))
    if smoke_probs.shape != expected_smoke:
        raise RuntimeError(
            f"Loaded classifier failed smoke check: expected {expected_smoke}, "
            f"got {smoke_probs.shape}."
        )
    print("Classifier smoke check:", smoke_probs.shape)

    (
        predictions,
        confusion,
        family_confusion,
        mean_probability,
        redraw_stats,
    ) = classify_fresh_data(
        approximator=approximator,
        n_per_model=cfg["n_per_model"],
        batch_size=cfg["batch_size"],
    )

    confusion_norm = row_normalize(confusion)
    family_confusion_norm = row_normalize(family_confusion)

    predictions.to_csv(
        recovery_dir / "model_recovery_predictions.csv",
        index=False,
    )

    save_matrix_csv(
        confusion,
        sim.MODEL_NAMES,
        sim.MODEL_NAMES,
        recovery_dir / "confusion_matrix_counts.csv",
    )
    save_matrix_csv(
        confusion_norm,
        sim.MODEL_NAMES,
        sim.MODEL_NAMES,
        recovery_dir / "confusion_matrix_normalized.csv",
    )
    save_matrix_csv(
        mean_probability,
        sim.MODEL_NAMES,
        sim.MODEL_NAMES,
        recovery_dir / "mean_model_probability_matrix.csv",
    )

    family_names = ["DFT", "RUDT"]
    save_matrix_csv(
        family_confusion,
        family_names,
        family_names,
        recovery_dir / "family_confusion_matrix_counts.csv",
    )
    save_matrix_csv(
        family_confusion_norm,
        family_names,
        family_names,
        recovery_dir / "family_confusion_matrix_normalized.csv",
    )

    per_model = make_per_model_metrics(predictions)
    per_model.to_csv(
        recovery_dir / "per_model_recovery_metrics.csv",
        index=False,
    )

    draw_heatmap(
        confusion,
        DISPLAY_NAMES,
        DISPLAY_NAMES,
        "8-model recovery: confusion matrix (counts)",
        fig_dir / "confusion_matrix_counts.png",
        "d",
    )
    draw_heatmap(
        confusion_norm,
        DISPLAY_NAMES,
        DISPLAY_NAMES,
        "8-model recovery: row-normalized confusion matrix",
        fig_dir / "confusion_matrix_normalized.png",
        ".2f",
        vmin=0,
        vmax=1,
    )
    draw_heatmap(
        mean_probability,
        DISPLAY_NAMES,
        DISPLAY_NAMES,
        "Mean posterior model probabilities by true model",
        fig_dir / "mean_model_probability_matrix.png",
        ".2f",
        vmin=0,
        vmax=1,
    )
    draw_heatmap(
        family_confusion_norm,
        family_names,
        family_names,
        "Family recovery: DFT vs RUDT",
        fig_dir / "family_confusion_matrix_normalized.png",
        ".3f",
        vmin=0,
        vmax=1,
    )

    overall_accuracy = float(predictions["correct"].mean())
    family_accuracy = float(predictions["family_correct"].mean())
    macro_recall = float(per_model["recall_exact_model"].mean())

    summary = {
        "mode": args.mode,
        "classifier": str(classifier_path),
        "model_names": list(sim.MODEL_NAMES),
        "n_trials": len(sim.DEMURIE_DESIGN),
        "n_per_model": cfg["n_per_model"],
        "n_total": len(predictions),
        "batch_size": cfg["batch_size"],
        "overall_exact_model_accuracy": overall_accuracy,
        "macro_exact_model_recall": macro_recall,
        "overall_family_accuracy": family_accuracy,
        "fresh_test_redraw_stats": redraw_stats,
    }
    (recovery_dir / "model_recovery_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("=" * 80)
    print("MODEL RECOVERY COMPLETE")
    print("classifier:", classifier_path)
    print("n per model:", cfg["n_per_model"])
    print("overall exact-model accuracy:", f"{overall_accuracy:.4f}")
    print("macro exact-model recall:", f"{macro_recall:.4f}")
    print("DFT/RUDT family accuracy:", f"{family_accuracy:.4f}")
    print("outputs:", recovery_dir)
    print("=" * 80)
    print(per_model.to_string(index=False))

    del approximator
    keras.backend.clear_session()
    cleanup()


if __name__ == "__main__":
    main()
