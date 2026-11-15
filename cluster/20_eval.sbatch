#!/bin/bash
# ---------------------------------------------------------------------------
# 20 — Evaluate checkpoints of one run: decisions, self-reports, controls.
#
#   sbatch --account="$TORCH_ACCOUNT" --chdir="$REPO" --array=1-4 cluster/20_eval.sbatch
#
# Defaults: base model (step 0) + every 400th checkpoint + the final one, 24 reports
# per character, observer baseline at the final checkpoint.  Override with EVAL_ARGS,
# e.g. EVAL_ARGS="--every 200 --observer-steps all".  Finished steps are skipped,
# so the job is resumable and can be rerun after training adds checkpoints.
# ---------------------------------------------------------------------------
#SBATCH --job-name=bi-eval
#SBATCH --account=torch_pr_1363_cds
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --constraint="h200"
#SBATCH --time=16:00:00
#SBATCH --output=logs/bi-eval-%A_%a.out
#SBATCH --error=logs/bi-eval-%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kk6081@nyu.edu
set -euo pipefail
source "${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}/cluster/job_common.sh"
resolve_levels
nvidia-smi || true
in_container "
  python -m boolintro.evaluate --run-dir $RUN_DIR ${EVAL_ARGS:---every 400}
  python -m boolintro.analyze --runs $RUN_DIR --out $RUN_DIR/analysis
"
