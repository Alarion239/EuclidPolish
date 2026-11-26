#!/bin/bash
#SBATCH --job-name=dr1-population-calibration
#SBATCH --partition=test
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=12:00:00
#SBATCH --output=logs/jobs/dr1-population-calibration-%j.out
#SBATCH --error=logs/jobs/dr1-population-calibration-%j.err
# Rerun the synthetic-sky population calibration on Euclid DR1 Wide (300-tile
# sample): galaxy counts/radii/colour+SFR catalogue, star counts/colours, and
# the MER noise-level table. Outputs go to $EUCLID_POLISH_DATA_DIR/population_comparison/dr1/
# and euclid_polish/sky/observation/mer_noise_levels_dr1.json; nothing is
# activated. Free `test` partition (billing weight 0, 12 h); every step resumes,
# so a timed-out run is continued by resubmitting with --steps <remaining>.
#
#   sbatch scripts/fasrc_dr1_population_calibration.sh [--steps a,b,...]
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
mkdir -p logs/jobs
export EUCLID_POLISH_DATA_DIR=/n/netscratch/lconnor_lab/Lab/abelotserkovtsev/EuclidPolish/data
export EUCLID_POLISH_CKPT_DIR=/n/netscratch/lconnor_lab/Lab/abelotserkovtsev/EuclidPolish/ckpt/wdsr
export EUCLID_POLISH_CALIBRATION_RELEASE=dr1
export TQDM_DISABLE=1
export CUDA_VISIBLE_DEVICES=""
module purge
module load python
CONDA_BASE="$(conda info --base 2>/dev/null || true)"
if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  source "$CONDA_BASE/etc/profile.d/conda.sh"
fi
if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/mamba.sh" ]; then
  source "$CONDA_BASE/etc/profile.d/mamba.sh"
fi
mamba activate /n/holylabs/lconnor_lab/Lab/abelotserkovtsev/conda-env
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
echo "Job id: ${SLURM_JOB_ID:-local}  Host: $(hostname)  Started: $(date)  Commit: $(git rev-parse --short HEAD)"
python -u scripts/run_dr1_population_calibration.py --keep-going "$@"
echo "Finished: $(date)"
