#!/bin/bash
# ---------------------------------------------------------------------------
# 41 — Combined conflict-test analysis (CPU only).  Submit after 40_conflict.sbatch:
#
#   sbatch --account="$TORCH_ACCOUNT" --chdir="$REPO" --dependency=afterok:<jobid> cluster/41_conflict_analyze.sbatch
#
# CONFLICT_LEVELS (default "1 2") picks the runs; output: $RUNS_ROOT/${TAG}_conflict_analysis/
# ---------------------------------------------------------------------------
#SBATCH --job-name=bi-conflict-an
#SBATCH --account=torch_pr_1363_cds
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/bi-conflict-an-%j.out
#SBATCH --error=logs/bi-conflict-an-%j.err
set -euo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$REPO/cluster/env.sh"
RUNS=""
for L in ${CONFLICT_LEVELS:-1 2}; do RUNS="$RUNS $RUNS_ROOT/${TAG}_L$L"; done
echo "runs: $RUNS"
srun apptainer exec --overlay "$OVERLAY:ro" "$SIF" /bin/bash -c "
  source /ext3/env.sh
  cd $REPO
  python -m boolintro.conflict_analyze --runs $RUNS --out $RUNS_ROOT/${TAG}_conflict_analysis
"
