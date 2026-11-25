#!/bin/bash
# ---------------------------------------------------------------------------
# 51 — How many reports does pretraining need?  Sweep N = number of class-1 characters
# whose rules are shown (formula reports), from-scratch GPT-2.  One seed per N by default.
#
#   sbatch --account="$TORCH_ACCOUNT" --chdir="$REPO" --array=0-5 cluster/51_tiny_sweep.sbatch
#
# Task i runs N = NS[i / N_SEEDS], seed = i % N_SEEDS (6 tasks for the default 6 values of N
# and N_SEEDS=1).  For error bars use N_SEEDS=3 with --array=0-17.
# Every N uses the same 1200 class-1 characters and the same 200 class-2 characters;
# only the first N class-1 characters keep their reports.  Each task: pretrain ->
# finetune --train embed -> probe.  Runs land in $RUNS_ROOT/tiny/sweep_n<N>_s<seed>.
# Override the grid with NS="0 3 10 30 100 300 1000 1200" (and a matching --array:
# number of N values x N_SEEDS tasks).
# Analysis (locally after sync_from_torch.sh):
#   python -m boolintro.tiny.sweep --runs runs/tiny/sweep_n*_s* --out runs/tiny_sweep
# ---------------------------------------------------------------------------
#SBATCH --job-name=bi-sweep
#SBATCH --account=torch_pr_1363_cds
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|h200"
#SBATCH --time=6:00:00
#SBATCH --output=logs/bi-sweep-%A_%a.out
#SBATCH --error=logs/bi-sweep-%A_%a.err
set -euo pipefail
source "${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}/cluster/job_common.sh"
read -r -a NARR <<< "${NS:-0 10 30 100 300 1000}"
I="${SLURM_ARRAY_TASK_ID:-0}"
K="${N_SEEDS:-1}"
N="${NARR[$((I / K))]}"
SEED=$((I % K))
RUN_DIR="$RUNS_ROOT/tiny/sweep_n${N}_s${SEED}"
echo "host=$(hostname) N=$N seed=$SEED run_dir=$RUN_DIR"
nvidia-smi || true
in_container "
  python -m boolintro.tiny.train pretrain --run-dir $RUN_DIR --seed $SEED --report-format formula --n-with-reports $N ${PRE_ARGS:-}
  python -m boolintro.tiny.train finetune --run-dir $RUN_DIR --seed $SEED --train embed ${FT_ARGS:-}
  python -m boolintro.tiny.probe --run-dir $RUN_DIR
"
