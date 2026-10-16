#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTORCH_MODULE="${PYTORCH_MODULE:-pytorch/2.8.0}"

echo "Repository: $ROOT"
echo "Loading OSC module: $PYTORCH_MODULE"

module reset
module load "$PYTORCH_MODULE"

export PYTHONNOUSERSITE=True
export KERAS_BACKEND=torch

PKG_DIR="$ROOT/.python_packages"
mkdir -p "$PKG_DIR"

echo "Python: $(which python)"
python --version
echo "Installing project-local packages into: $PKG_DIR"

python -m pip install --upgrade --target "$PKG_DIR" -r requirements.txt

export PYTHONPATH="$PKG_DIR${PYTHONPATH:+:$PYTHONPATH}"

echo
echo "Verifying imports..."
python - <<'PY'
from importlib.metadata import version
import bayesflow
import keras
import torch
import ssms
import numpy
import scipy
import pandas
import matplotlib

print("BayesFlow:", version("bayesflow"))
print("Keras:", version("keras"))
print("Keras backend:", keras.backend.backend())
print("PyTorch:", torch.__version__)
print("CUDA available on this node:", torch.cuda.is_available())
print("ssms:", version("ssms"))
print("NumPy:", numpy.__version__)
print("SciPy:", scipy.__version__)
print("pandas:", pandas.__version__)
print("matplotlib:", matplotlib.__version__)

assert version("bayesflow") == "2.0.13", "BayesFlow version mismatch"
assert keras.backend.backend() == "torch", "Keras backend is not torch"
PY

echo
echo "OSC setup complete."
echo "Next run:"
echo "  sbatch verify_gpu.sbatch"
echo "then:"
echo "  sbatch train.sbatch --model rudt_sym_fixed --mode test"
