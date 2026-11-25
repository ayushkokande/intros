#!/bin/bash
# ---------------------------------------------------------------------------
# 30 — Combine all runs of a TAG into one report and figures (CPU only).
#
#   sbatch --account="$TORCH_ACCOUNT" --chdir="$REPO" cluster/30_analyze.sbatch
# Output: $RUNS_ROOT/${TAG}_analysis/{report.md,summary.csv,*.png}
# ---------------------------------------------------------------------------
#SBATCH --job-name=bi-analyze
#SBATCH --account=torch_pr_1363_cds
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/bi-analyze-%j.out
#SBATCH --error=logs/bi-analyze-%j.err
set -euo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$REPO/cluster/env.sh"
RUNS=$(ls -d "$RUNS_ROOT/${TAG}"_L* 2>/dev/null | tr '\n' ' ')
[ -n "$RUNS" ] || { echo "no runs matching $RUNS_ROOT/${TAG}_L*" >&2; exit 1; }
echo "runs: $RUNS"
srun apptainer exec --overlay "$OVERLAY:ro" "$SIF" /bin/bash -c "
  source /ext3/env.sh
  cd $REPO
  python -m boolintro.analyze --runs $RUNS --out $RUNS_ROOT/${TAG}_analysis
"
