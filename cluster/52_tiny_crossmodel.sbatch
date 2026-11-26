#!/bin/bash
# ---------------------------------------------------------------------------
# 52 — Cross-model control on the from-scratch GPT-2 (boolintro/tiny/crossmodel.py).
# One array task per seed: pretrain A (self), A2 (outsider "same": same size, new init) and
# B (outsider "other": different size), finetune, swap roles, evaluate.
#
#   sbatch --account="$TORCH_ACCOUNT" --chdir="$REPO" --array=0-2 cluster/52_tiny_crossmodel.sbatch
#   # then (any node, CPU is fine):
#   bash cluster/run.sh python -m boolintro.crossmodel summarize \
#       --exps $RUNS_ROOT/tiny_xm/s*/self_* --out $RUNS_ROOT/tiny_xm/summary
#
# Defaults follow the main tiny sandbox (6 x 256 GPT-2, 1000 report characters, 30k steps);
# B is 3 x 384.  Override all of it with TINY_XM_ARGS.  Rerunning skips finished stages.
# ---------------------------------------------------------------------------
#SBATCH --job-name=bi-tiny-xm
#SBATCH --account=torch_pr_1363_cds
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|h200"
#SBATCH --time=12:00:00
#SBATCH --output=logs/bi-tiny-xm-%A_%a.out
#SBATCH --error=logs/bi-tiny-xm-%A_%a.err
set -euo pipefail
source "${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}/cluster/job_common.sh"
SEED="${SLURM_ARRAY_TASK_ID:-0}"
EXP="$RUNS_ROOT/tiny_xm/s${SEED}"
ARGS="${TINY_XM_ARGS:---n-pre-report 1000 --n-new 200 --pretrain-steps 30000 --finetune-steps 5000 \
  --n-layer 6 --n-embd 256 --n-head 8 --other-n-layer 3 --other-n-embd 384 --other-n-head 8 --n-samples 8}"
echo "host=$(hostname) seed=$SEED exp=$EXP"
nvidia-smi || true
in_container "python -m boolintro.tiny.crossmodel run --exp $EXP --seed $SEED $ARGS"
