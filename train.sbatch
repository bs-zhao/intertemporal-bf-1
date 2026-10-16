#!/bin/bash
#SBATCH --account=PDE0114
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --job-name=bf8-param
#SBATCH --output=slurm-%j.out

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
module reset
module load pytorch/2.8.0

export PYTHONNOUSERSITE=True
export KERAS_BACKEND=torch
export PYTHONPATH="$SLURM_SUBMIT_DIR/.python_packages${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -d "$SLURM_SUBMIT_DIR/.python_packages" ]]; then
    echo "ERROR: .python_packages is missing."
    echo "Run 'bash setup_osc.sh' once from the repository root before submitting jobs."
    exit 1
fi

echo "Host: $(hostname)"
echo "Working directory: $PWD"
echo "Python: $(which python)"
python --version
python - <<'PY'
from importlib.metadata import version
import torch, keras
print("BayesFlow:", version("bayesflow"))
print("Keras backend:", keras.backend.backend())
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA devices:", torch.cuda.device_count())
if torch.cuda.device_count() > 0:
    print("GPU:", torch.cuda.get_device_name(0))
PY

python s1_train_parameter_posterior.py "$@"
