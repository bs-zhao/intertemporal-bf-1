#!/bin/bash
#SBATCH --account=PDE0114
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --job-name=bf8-recovery
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

python s2_parameter_recovery.py "$@"
