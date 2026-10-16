#!/bin/bash
#SBATCH --account=PDE0114
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --job-name=bf8-verify
#SBATCH --output=slurm-verify-%j.out

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
module reset
module load pytorch/2.8.0

export PYTHONNOUSERSITE=True
export KERAS_BACKEND=torch
export PYTHONPATH="$SLURM_SUBMIT_DIR/.python_packages${PYTHONPATH:+:$PYTHONPATH}"

python - <<'PY'
from importlib.metadata import version
import bayesflow
import keras
import torch
import ssms

print("BayesFlow:", version("bayesflow"))
print("Keras:", version("keras"))
print("Keras backend:", keras.backend.backend())
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())

if torch.cuda.device_count() > 0:
    print("GPU:", torch.cuda.get_device_name(0))
else:
    raise RuntimeError("No CUDA GPU is visible inside the requested GPU job.")

assert version("bayesflow") == "2.0.13"
assert keras.backend.backend() == "torch"
print("Environment verification passed.")
PY
