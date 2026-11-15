#!/bin/bash
# ---------------------------------------------------------------------------
# 03 — Smoke test: unit tests + a tiny end-to-end run with the small model.
#
#   source /scratch/$USER/intros/cluster/env.sh
#   sbatch --account="$TORCH_ACCOUNT" --chdir="$REPO" cluster/03_smoke.sbatch
#
# Trains 60 steps on 10 level-1 characters, evaluates base + 2 checkpoints with
# the observer, and runs the analysis.  Checks the plumbing, not the science.
# ---------------------------------------------------------------------------
#SBATCH --job-name=bi-smoke
#SBATCH --account=torch_pr_1363_cds
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|h200"
#SBATCH --time=00:45:00
#SBATCH --output=logs/bi-smoke-%j.out
#SBATCH --error=logs/bi-smoke-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kk6081@nyu.edu
set -euo pipefail
source "${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}/cluster/job_common.sh"
SMOKE_DIR="$RUNS_ROOT/smoke_$SLURM_JOB_ID"
nvidia-smi || true
in_container "
  python -m pytest -q tests
  python -m boolintro.train --run-dir $SMOKE_DIR/L1 --model $SMOKE_MODEL --levels 1 \
      --n-characters 10 --steps 60 --batch-size 16 --micro-batch 16 --warmup 10 --lr 2e-4 \
      --save-every 30 --log-every 10
  python -m boolintro.evaluate --run-dir $SMOKE_DIR/L1 --n-reports 4 --observer-k 8 --observer-n 2
  python -m boolintro.analyze --runs $SMOKE_DIR/L1 --out $SMOKE_DIR/analysis
"
echo "smoke OK: $SMOKE_DIR"
