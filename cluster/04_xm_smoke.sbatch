#!/bin/bash
# ---------------------------------------------------------------------------
# 04 — Smoke test of the cross-model control with small models (about 20-30 minutes, any GPU).
#
#   EXTRA_MODELS=HuggingFaceTB/SmolLM2-360M-Instruct bash $REPO/cluster/02_prefetch.sh   # login node
#   sbatch --account="$TORCH_ACCOUNT" --chdir="$REPO" cluster/04_xm_smoke.sbatch
#
# Self = $SMOKE_MODEL (Qwen3-0.6B) on 10 level-1 characters; outsiders = a fresh Qwen3-0.6B
# LoRA (same) and SmolLM2-360M (other, a Llama-architecture model that needs no licence).
# Checks the plumbing, not the science.
# ---------------------------------------------------------------------------
#SBATCH --job-name=bi-xm-smoke
#SBATCH --account=torch_pr_1363_cds
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/bi-xm-smoke-%j.out
#SBATCH --error=logs/bi-xm-smoke-%j.err
set -euo pipefail
source "${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}/cluster/job_common.sh"
D="$RUNS_ROOT/xm_smoke_$SLURM_JOB_ID"
SMOKE_OTHER="${SMOKE_OTHER:-HuggingFaceTB/SmolLM2-360M-Instruct}"
T="--n-characters 10 --steps 200 --batch-size 16 --micro-batch 16 --warmup 20 --lr 2e-4 --save-every 100"
nvidia-smi || true
in_container "
  python -m pytest -q tests/test_crossmodel.py
  python -m boolintro.train --run-dir $D/m1 --model $SMOKE_MODEL --levels 1 $T
  python -m boolintro.crossmodel record --m1-run $D/m1 --exp $D/xm
  python -m boolintro.train --run-dir $D/xm/m2_same --model $SMOKE_MODEL --init-seed 1 \
      --spec-from $D/xm/spec.json --labels-from $D/xm/m1_choices.json $T
  python -m boolintro.train --run-dir $D/xm/m2_other --model $SMOKE_OTHER \
      --spec-from $D/xm/spec.json --labels-from $D/xm/m1_choices.json $T
  python -m boolintro.crossmodel evaluate --exp $D/xm --m2-run $D/xm/m2_same --n-reports 8
  python -m boolintro.crossmodel evaluate --exp $D/xm --m2-run $D/xm/m2_other --n-reports 8
  python -m boolintro.crossmodel summarize --exps $D/xm --out $D/summary
"
echo "smoke OK: $D/summary/report.md"
