#!/bin/bash
#SBATCH --job-name=euclid-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# -----------------------------------------------------------------------------
# Harvard FASRC Cannon — full EuclidPolish pipeline:
#   generate 6400 train + 200 validate clean fields (512² HR @ 0.05"/pix)
# → convolve to 256² LR with realistic Euclid VIS noise
# → train WDSR for 400 000 iterations
#
# Submit from the project root:
#     sbatch scripts/fasrc_train.sh
#
# CPU request: training itself is GPU-bound — the actual gradient computation
# is a single TF graph that runs on the GPU. The cpus-per-task=8 above is
# enough to keep `tf.data` (TFRecord parsing + augmentation, parallelized via
# AUTOTUNE) and async prefetch from starving the GPU. Going much higher
# wastes scheduler resources without speeding training. The clean-field
# generation stage *is* effectively single-threaded (GalSim galaxy drawing),
# so giving it more cores wouldn't help either.
# -----------------------------------------------------------------------------

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_ROOT"
# logs/ is committed (with .gitkeep) so SLURM has a place to write %x-%j.out
# from the very first scheduling attempt. mkdir below is a safety net only.
mkdir -p logs

echo "============================================================"
echo "Job:       ${SLURM_JOB_ID:-local}"
echo "Host:      $(hostname)"
echo "Started:   $(date)"
echo "Workdir:   $(pwd)"
echo "GPUs:"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv 2>/dev/null || true
echo "============================================================"

# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------
module purge
module load python
module load cuda

# `mamba activate` needs the conda/mamba shell hook in non-interactive bash.
# `module load python` on Cannon ships Mambaforge and registers the hook, but
# we source it explicitly so the script is robust across module versions.
if [ -z "${CONDA_SHLVL:-}" ]; then
  CONDA_BASE="$(conda info --base 2>/dev/null || true)"
  if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "$CONDA_BASE/etc/profile.d/conda.sh"
  fi
  if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/mamba.sh" ]; then
    # shellcheck disable=SC1091
    source "$CONDA_BASE/etc/profile.d/mamba.sh"
  fi
fi

mamba activate /n/holylabs/lconnor_lab/Lab/abelotserkovtsev

echo "Python:    $(which python)"
echo "Python v:  $(python -V 2>&1)"
echo "CUDA dev:  ${CUDA_VISIBLE_DEVICES:-unset}"
echo "============================================================"

# -----------------------------------------------------------------------------
# Pipeline
# -----------------------------------------------------------------------------
python -u scripts/run_pipeline.py \
  --ntrain 6400 \
  --nvalid 200 \
  --image-size 512 \
  --batch-size 16 \
  --steps 400000

echo "============================================================"
echo "Finished:  $(date)"
echo "============================================================"
