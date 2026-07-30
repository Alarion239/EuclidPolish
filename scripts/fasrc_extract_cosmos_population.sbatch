#!/usr/bin/env bash
#SBATCH --job-name=cosmos_population
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/cosmos_population_%j.out
#SBATCH --error=logs/cosmos_population_%j.err

set -euo pipefail

EUCLID_REPO_PATH="${EUCLID_POLISH_REPO_PATH:-/n/holylabs/lconnor_lab/Lab/abelotserkovtsev/EuclidPolish}"
EUCLID_DATA_PATH="${EUCLID_POLISH_DATA_DIR:-/n/netscratch/lconnor_lab/Lab/abelotserkovtsev/EuclidPolish/data}"
EUCLID_PYTHON_BIN="${EUCLID_POLISH_PYTHON:-/n/holylabs/lconnor_lab/Lab/abelotserkovtsev/conda-env/bin/python}"

cd "$EUCLID_REPO_PATH"
mkdir -p logs "$EUCLID_DATA_PATH/population_comparison/cosmos2025"
export EUCLID_POLISH_DATA_DIR="$EUCLID_DATA_PATH"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/euclid_mpl_cache"

echo "COSMOS2025 population-prior extraction"
echo "Job id:   ${SLURM_JOB_ID:-manual}"
echo "Host:     $(hostname)"
echo "Python:   $EUCLID_PYTHON_BIN"
echo "Catalog:  $EUCLID_DATA_PATH/COSMOS2025/cosmos2025.fits"
echo "Output:   $EUCLID_DATA_PATH/population_comparison/cosmos2025"

"$EUCLID_PYTHON_BIN" -u scripts/fasrc_extract_cosmos_population.py \
  --catalog "$EUCLID_DATA_PATH/COSMOS2025/cosmos2025.fits" \
  --out-dir "$EUCLID_DATA_PATH/population_comparison/cosmos2025" \
  --area-deg2 0.54 \
  "$@"
