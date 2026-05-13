#!/bin/bash
#SBATCH --job-name=euclid-train-only
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# -----------------------------------------------------------------------------
# Train-only variant: skips generate + forward, assumes the TFRecords
# already exist under ``$EUCLID_POLISH_DATA_DIR/images/records_v2/``:
#   clean_{train,validate}.tfrecord  — 4-band HR clean (inspection)
#   dirty_{train,validate}.tfrecord  — 4-band LR dirty (model input)
#   hr_{train,validate}.tfrecord     — 1-band VIS HR (training target)
#
# Submit from the project root:
#     sbatch scripts/fasrc_train_only.sh
#
# Memory: 32G — no COSMOS catalog in RAM, no convolution buffers.
# CPUs:   8  — sufficient for tf.data pipeline parallelism.
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
# Data + checkpoint locations (must match the prior fasrc_train.sh run)
# -----------------------------------------------------------------------------
export EUCLID_POLISH_DATA_DIR=/n/netscratch/lconnor_lab/Lab/abelotserkovtsev/EuclidPolish/data
export EUCLID_POLISH_CKPT_DIR=/n/netscratch/lconnor_lab/Lab/abelotserkovtsev/EuclidPolish/ckpt/wdsr
mkdir -p "$EUCLID_POLISH_CKPT_DIR"
echo "Data root:     $EUCLID_POLISH_DATA_DIR"
echo "Checkpoints:   $EUCLID_POLISH_CKPT_DIR"

module purge
module load python
module load cuda

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

# Sanity-check that the tfrecords from a prior generate+forward run exist.
REC="$EUCLID_POLISH_DATA_DIR/images/records_v2"
for f in clean_train.tfrecord clean_validate.tfrecord \
         dirty_train.tfrecord dirty_validate.tfrecord \
         hr_train.tfrecord    hr_validate.tfrecord ; do
  if [ ! -f "$REC/$f" ]; then
    echo "MISSING: $REC/$f"
    echo "Run scripts/fasrc_train.sh once first to generate + forward the dataset."
    exit 2
  fi
done
echo "Existing tfrecords:"
ls -lh "$REC"/{clean,dirty,hr}_{train,validate}.tfrecord
echo "============================================================"

python -u scripts/run_pipeline.py \
  --skip-generate \
  --skip-convolve \
  --batch-size 16 \
  --steps 400000

echo "============================================================"
echo "Finished:  $(date)"
echo "============================================================"
