#!/usr/bin/env python

"""
Train one BayesFlow 8-class model-comparison classifier.

The eight competing models are the exact models in simulators_8models.py.
Training uses fresh prior-predictive simulations from the shared 96-trial
Demurie design and a uniform model prior.

Crucially, the best classifier is saved DURING training. Every time val_loss
improves, the full BayesFlow/Keras approximator is first written to a temporary
.keras file and then atomically moved to classifier_best.keras. Therefore
s5_model_recovery.py can use the latest complete best checkpoint even when this
training script has not finished.

Examples
--------
python s4_train_model_classifier.py --mode test
python s4_train_model_classifier.py --mode formal
"""

import os
os.environ.setdefault("KERAS_BACKEND", "torch")

import argparse
import json
from pathlib import Path
from importlib.metadata import version as package_version

import numpy as np
import pandas as pd
import keras
import bayesflow as bf

import simulators_8models as sim


BF_REQUIRED_VERSION = "2.0.13"
BASE_SEED = 20261016

INITIAL_LR = 5e-4
MIN_LR = 1e-6

FORMAL_TRAIN_CFG = dict(
    epochs=5000,
    num_batches_per_epoch=50,
    batch_size=64,
    validation_data=2000,
)

TEST_TRAIN_CFG = dict(
    epochs=5,
    num_batches_per_epoch=50,
    batch_size=64,
    validation_data=2000,
)

OUTPUT_ROOT = Path("output") / "model_recovery"


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

    try:
        import torch
        print("PyTorch:", torch.__version__)
        print("CUDA available:", torch.cuda.is_available())
        print("CUDA device count:", torch.cuda.device_count())
        if torch.cuda.device_count() > 0:
            print("GPU:", torch.cuda.get_device_name(0))
    except Exception as exc:
        print("Torch device inspection warning:", repr(exc))

    print("BayesFlow:", v)
    print("Keras backend:", keras.backend.backend())
    print("Models:", len(sim.MODEL_NAMES))
    print("Demurie trials:", len(sim.DEMURIE_DESIGN))


def make_model_only_simulator(model, seed):
    """Sample parameters internally and return only the observable trial set."""
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


def build_model_comparison_simulator(seed):
    simulators = []
    redraw_stats = {}

    for i, model in enumerate(sim.MODEL_NAMES):
        model_sim, stats = make_model_only_simulator(
            model=model,
            seed=seed + 10000 * (i + 1),
        )
        simulators.append(model_sim)
        redraw_stats[model] = stats

    mc_simulator = bf.simulators.ModelComparisonSimulator(
        simulators=simulators,
        use_mixed_batches=True,
    )
    return mc_simulator, redraw_stats


def make_adapter():
    return (
        bf.Adapter()
        .convert_dtype("float64", "float32")
        .rename("trials", "summary_variables")
        .rename("model_indices", "inference_variables")
    )


def make_optimizer():
    return keras.optimizers.Adam(
        learning_rate=INITIAL_LR,
        clipnorm=1.5,
    )


def build_workflow(mc_simulator):
    return bf.ModelComparisonWorkflow(
        simulator=mc_simulator,
        adapter=make_adapter(),
        summary_network=bf.networks.SetTransformer(summary_dim=32),
        model_names=list(sim.MODEL_NAMES),
        scoring_rules="cross_entropy",
        optimizer=make_optimizer(),
    )


class AtomicBestModelSaver(keras.callbacks.Callback):
    """Atomically maintain the full best .keras classifier during training."""

    def __init__(self, filepath, status_path, monitor="val_loss", mode="min"):
        super().__init__()
        self.filepath = Path(filepath)
        self.status_path = Path(status_path)
        self.monitor = monitor
        self.mode = mode
        self.best = np.inf if mode == "min" else -np.inf
        self.best_epoch = None

    def _improved(self, value):
        if self.mode == "min":
            return value < self.best
        return value > self.best

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        value = logs.get(self.monitor)
        if value is None:
            return

        value = float(value)
        if not np.isfinite(value) or not self._improved(value):
            return

        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        tmp_model = self.filepath.with_name(
            self.filepath.stem + ".tmp" + self.filepath.suffix
        )
        tmp_status = self.status_path.with_name(
            self.status_path.stem + ".tmp" + self.status_path.suffix
        )

        if tmp_model.exists():
            tmp_model.unlink()

        self.model.save(tmp_model, overwrite=True)
        os.replace(tmp_model, self.filepath)

        status = {
            "best_epoch": int(epoch) + 1,
            "best_val_loss": value,
            "classifier_best": str(self.filepath),
        }
        tmp_status.write_text(json.dumps(status, indent=2), encoding="utf-8")
        os.replace(tmp_status, self.status_path)

        self.best = value
        self.best_epoch = int(epoch) + 1
        print(
            f"Atomic best checkpoint updated: epoch={self.best_epoch}, "
            f"val_loss={self.best:.6f}, path={self.filepath}"
        )


def make_callbacks(best_path, status_path, live_history_path):
    return [
        AtomicBestModelSaver(
            filepath=best_path,
            status_path=status_path,
            monitor="val_loss",
            mode="min",
        ),
        keras.callbacks.CSVLogger(
            filename=str(live_history_path),
            append=False,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=8,
            min_delta=1e-4,
            cooldown=0,
            min_lr=MIN_LR,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=20,
            min_delta=1e-4,
            restore_best_weights=True,
            verbose=1,
        ),
    ]


def parse_args():
    p = argparse.ArgumentParser(
        description="Train the 8-class BayesFlow model-comparison classifier."
    )
    p.add_argument("--mode", required=True, choices=("test", "formal"))
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing classifier_best.keras.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    check_environment()

    cfg = TEST_TRAIN_CFG.copy() if args.mode == "test" else FORMAL_TRAIN_CFG.copy()
    run_dir = OUTPUT_ROOT / args.mode
    run_dir.mkdir(parents=True, exist_ok=True)

    best_path = run_dir / "classifier_best.keras"
    final_path = run_dir / "classifier_final.keras"
    status_path = run_dir / "best_checkpoint.json"
    live_history_path = run_dir / "training_history_live.csv"

    if best_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{best_path} already exists. Use --overwrite only if you intentionally "
            "want to start a new classifier training run."
        )

    for p in (best_path, final_path, status_path, live_history_path):
        if args.overwrite and p.exists():
            p.unlink()

    keras.utils.set_random_seed(BASE_SEED)

    mc_simulator, redraw_stats = build_model_comparison_simulator(BASE_SEED)

    smoke = mc_simulator.sample(16)
    print("Raw model-comparison smoke shapes:")
    print({k: np.asarray(v).shape for k, v in smoke.items()})

    expected_model_shape = (16, len(sim.MODEL_NAMES))
    got_model_shape = np.asarray(smoke["model_indices"]).shape
    if got_model_shape != expected_model_shape:
        raise RuntimeError(
            f"Expected model_indices shape {expected_model_shape}, "
            f"got {got_model_shape}."
        )

    workflow = build_workflow(mc_simulator)
    adapted = workflow.adapter(smoke)
    adapted_shapes = {k: np.asarray(v).shape for k, v in adapted.items()}
    print("Adapted shapes:", adapted_shapes)

    expected_summary_shape = (
        16,
        len(sim.DEMURIE_DESIGN),
        len(sim.TRIAL_FEATURE_NAMES),
    )
    if adapted_shapes.get("summary_variables") != expected_summary_shape:
        raise RuntimeError(
            f"Expected summary_variables shape {expected_summary_shape}, "
            f"got {adapted_shapes.get('summary_variables')}."
        )
    if adapted_shapes.get("inference_variables") != expected_model_shape:
        raise RuntimeError(
            f"Expected inference_variables shape {expected_model_shape}, "
            f"got {adapted_shapes.get('inference_variables')}."
        )

    metadata = {
        "mode": args.mode,
        "model_names": list(sim.MODEL_NAMES),
        "num_models": len(sim.MODEL_NAMES),
        "model_prior": "uniform",
        "n_trials": len(sim.DEMURIE_DESIGN),
        "trial_feature_names": list(sim.TRIAL_FEATURE_NAMES),
        "seed": BASE_SEED,
        "bayesflow_version": package_version("bayesflow"),
        "keras_backend": keras.backend.backend(),
        "training_config": cfg,
        "initial_lr": INITIAL_LR,
        "min_lr": MIN_LR,
        "summary_network": "SetTransformer(summary_dim=32)",
        "scoring_rules": "cross_entropy",
        "best_checkpoint": str(best_path),
        "checkpoint_policy": "atomic save whenever val_loss improves",
    }
    (run_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("=" * 80)
    print("8-MODEL CLASSIFIER TRAINING")
    print("mode:", args.mode)
    print("config:", cfg)
    print("best checkpoint:", best_path)
    print("=" * 80)

    callbacks = make_callbacks(
        best_path=best_path,
        status_path=status_path,
        live_history_path=live_history_path,
    )

    history = workflow.fit_online(
        **cfg,
        callbacks=callbacks,
    )

    history_df = pd.DataFrame(history.history)
    history_df.to_csv(run_dir / "training_history.csv", index=False)

    workflow.approximator.save(final_path)

    best_status = {}
    if status_path.exists():
        best_status = json.loads(status_path.read_text(encoding="utf-8"))

    summary = {
        **metadata,
        "epochs_completed": len(history_df),
        "best_epoch": best_status.get("best_epoch"),
        "best_val_loss": best_status.get("best_val_loss"),
        "classifier_final": str(final_path),
        "redraw_stats": redraw_stats,
    }
    (run_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("Training complete.")
    print("Best classifier:", best_path)
    print("Final classifier:", final_path)
    print("Best status:", best_status)


if __name__ == "__main__":
    main()
