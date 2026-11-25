#!/bin/bash
# ---------------------------------------------------------------------------
# 40 — Conflict test: trained rule (weights) vs. forged record of past choices (context).
#
#   sbatch --account="$TORCH_ACCOUNT" --chdir="$REPO" --array=1-2 cluster/40_conflict.sbatch
#
# Runs boolintro.conflict on the final checkpoint of ${TAG}_L<level> (needs that run's
# eval/step_XXXXXX.json for character selection), for both the fine-tuned model and the
# base model (adapter off).  Resumable: rerun the same command after a timeout.
# Override with CONFLICT_ARGS, e.g. CONFLICT_ARGS="--ks 4,8,16,32 --min-faith 0.75".
# The combined analysis (both levels) is in 41_conflict_analyze.sbatch, or run locally
# after sync_from_torch.sh:  python -m boolintro.conflict_analyze --runs runs/qwen3-32b_L1 runs/qwen3-32b_L2 --out runs/conflict_analysis
# ---------------------------------------------------------------------------
#SBATCH --job-name=bi-conflict
#SBATCH --account=torch_pr_1363_cds
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --constraint="h200"
#SBATCH --time=16:00:00
#SBATCH --output=logs/bi-conflict-%A_%a.out
#SBATCH --error=logs/bi-conflict-%A_%a.err
set -euo pipefail
source "${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}/cluster/job_common.sh"
resolve_levels
nvidia-smi || true
in_container "
  python -m boolintro.conflict --run-dir $RUN_DIR ${CONFLICT_ARGS:-}
"
