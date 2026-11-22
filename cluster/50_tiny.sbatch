#!/bin/bash
# ---------------------------------------------------------------------------
# 50 — From-scratch small GPT-2 sandbox (boolintro/tiny): pretrain, finetune, probe.
#
# One array task per seed; WORLD picks the pretraining report format:
#   sbatch --account="$TORCH_ACCOUNT" --chdir="$REPO" --array=0-2 --export=ALL,WORLD=formula   cluster/50_tiny.sbatch
#   sbatch --account="$TORCH_ACCOUNT" --chdir="$REPO" --array=0-2 --export=ALL,WORLD=attribute cluster/50_tiny.sbatch
#
# Formula reports only for L1/L2 people in pretraining (a separate run name via TINY_TAG;
# set the variables in the environment, since --export splits on commas):
#   TINY_TAG=formula-L12reports PRE_ARGS="--report-families L1,L2" \
#     sbatch --account="$TORCH_ACCOUNT" --chdir="$REPO" --array=0-2 --export=ALL,WORLD=formula cluster/50_tiny.sbatch
#
# Runs land in $RUNS_ROOT/tiny/${TINY_TAG:-$WORLD}_s<seed>.  Each task runs
#   pretrain (all families by default) -> finetune --train embed -> finetune --train all
#   --replay 0.5 -> probe.
# Extra flags: PRE_ARGS (pretrain), FT_ARGS (both finetunes).  Every stage is skipped when
# already DONE, so a rerun after a timeout only redoes the unfinished stage.
# Analysis (CPU, or locally after sync_from_torch.sh):
#   python -m boolintro.tiny.analyze --runs runs/tiny/formula_s* runs/tiny/attribute_s* --out runs/tiny_analysis
# ---------------------------------------------------------------------------
#SBATCH --job-name=bi-tiny
#SBATCH --account=torch_pr_1363_cds
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|h200"
#SBATCH --time=8:00:00
#SBATCH --output=logs/bi-tiny-%A_%a.out
#SBATCH --error=logs/bi-tiny-%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kk6081@nyu.edu
set -euo pipefail
source "${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}/cluster/job_common.sh"
WORLD="${WORLD:-formula}"
SEED="${SLURM_ARRAY_TASK_ID:-0}"
RUN_DIR="$RUNS_ROOT/tiny/${TINY_TAG:-$WORLD}_s${SEED}"
echo "host=$(hostname) world=$WORLD seed=$SEED run_dir=$RUN_DIR"
nvidia-smi || true
in_container "
  python -m boolintro.tiny.train pretrain --run-dir $RUN_DIR --seed $SEED --report-format $WORLD ${PRE_ARGS:-}
  python -m boolintro.tiny.train finetune --run-dir $RUN_DIR --seed $SEED --train embed ${FT_ARGS:-}
  python -m boolintro.tiny.train finetune --run-dir $RUN_DIR --seed $SEED --train all --replay 0.5 --lr 3e-4 ${FT_ARGS:-}
  python -m boolintro.tiny.probe --run-dir $RUN_DIR
"
