#!/bin/bash
# ---------------------------------------------------------------------------
# 10 — Train one multi-character LoRA adapter per difficulty level.
#
# One array task per level (100 characters each):
#   sbatch --account="$TORCH_ACCOUNT" --chdir="$REPO" --array=1-4 cluster/10_train.sbatch
# A mixed run (25 characters per level, one adapter):
#   sbatch --account="$TORCH_ACCOUNT" --chdir="$REPO" --export=ALL,LEVELS=1,2,3,4 cluster/10_train.sbatch
# Smaller model (fits on L40S):
#   sbatch ... --array=1-4 --constraint="l40s|h200" \
#          --export=ALL,MODEL=Qwen/Qwen3-8B,TAG=qwen3-8b cluster/10_train.sbatch
#
# Extra train.py flags go in TRAIN_ARGS, e.g. TRAIN_ARGS="--steps 6000 --scheduler cosine".
# Resumable: on preemption or wall-time, resubmit the same command.
# Qwen3-32B in bf16 (~65 GB) + LoRA needs an H200; use --export=...,TRAIN_ARGS="--quant 8bit"
# with bitsandbytes installed to try other GPUs.
# ---------------------------------------------------------------------------
#SBATCH --job-name=bi-train
#SBATCH --account=torch_pr_1363_cds
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --constraint="h200"
#SBATCH --time=24:00:00
#SBATCH --output=logs/bi-train-%A_%a.out
#SBATCH --error=logs/bi-train-%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kk6081@nyu.edu
set -euo pipefail
source "${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}/cluster/job_common.sh"
resolve_levels
nvidia-smi || true
in_container "
  python -m boolintro.train --run-dir $RUN_DIR --model $MODEL --levels $LEVELS ${TRAIN_ARGS:-}
"
