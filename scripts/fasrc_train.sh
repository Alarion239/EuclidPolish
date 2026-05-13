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
# Harvard FASRC Cannon — full EuclidPolish multi-band pipeline:
#   1) generate 6400 train + 200 validate clean 4-band HR fields
#      (510² @ 0.05"/pix from the COSMOS2025 catalog)
#   2) forward model HR → 4-band LR (per-band PSF + Poisson + read noise +
#      cosmic-ray and hot-pixel artefacts)
#   3) train WDSR for 400 000 steps (4-channel LR → 1-channel VIS HR)
#
# Submit from the project root:
#     sbatch scripts/fasrc_train.sh
#
# CPU request: training itself is GPU-bound — the gradient step is a single
# TF graph that runs on the GPU. ``cpus-per-task=8`` is enough to keep
# ``tf.data`` (TFRecord parsing + augmentation, parallelised via AUTOTUNE)
# and async prefetch from starving the GPU.
# -----------------------------------------------------------------------------

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_ROOT"
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
# Data + checkpoint locations
# -----------------------------------------------------------------------------
# Heavy artefacts (catalog, cutouts, TFRecords, checkpoints) live on
# netscratch — fast Lustre Tier 0, regenerable, 90-day idle purge is OK
# while training is active. Source code stays on holylabs alongside this
# script. ``Config`` reads these env vars, so every constant resolves to
# the netscratch root with no per-script CLI plumbing needed.
export EUCLID_POLISH_DATA_DIR=/n/netscratch/lconnor_lab/Lab/abelotserkovtsev/EuclidPolish/data
export EUCLID_POLISH_CKPT_DIR=/n/netscratch/lconnor_lab/Lab/abelotserkovtsev/EuclidPolish/ckpt/wdsr
mkdir -p "$EUCLID_POLISH_DATA_DIR" "$EUCLID_POLISH_CKPT_DIR"
echo "Data root:     $EUCLID_POLISH_DATA_DIR"
echo "Checkpoints:   $EUCLID_POLISH_CKPT_DIR"

# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------
module purge
module load python
module load cuda

# ``mamba activate`` needs the conda/mamba shell hook in non-interactive bash.
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

mamba activate /n/holylabs/lconnor_lab/Lab/abelotserkovtsev/conda-env

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
  --image-size 510 \
  --batch-size 16 \
  --steps 400000

echo "============================================================"
echo "Finished:  $(date)"
echo "============================================================"
