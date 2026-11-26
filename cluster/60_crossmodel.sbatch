#!/bin/bash
# ---------------------------------------------------------------------------
# 60 — Cross-model control ("Looking Inward"), one array task per level.
# Normally submitted by cluster/submit_crossmodel.sh, which chains the stages.
#
# Self model M1 = $RUNS_ROOT/${TAG}_L<level> (a finished 10_train run).  XM_STAGE picks the stage:
#   record   M1's choices on train + held-out pairs        -> ${TAG}_L<level>_xm/m1_choices.json
#   same     outsider = fresh LoRA on M1's base model (--init-seed $SAME_INIT_SEED), train + evaluate
#   other    outsider = $OTHER_MODEL (a different family), train + evaluate
#   summary  side-by-side table of every *_xm experiment   -> $RUNS_ROOT/xm_summary/report.md
#
#   sbatch --account="$TORCH_ACCOUNT" --chdir="$REPO" --array=1-4 --export=ALL,XM_STAGE=record cluster/60_crossmodel.sbatch
#   sbatch ... --array=1-4 --export=ALL,XM_STAGE=same cluster/60_crossmodel.sbatch
#   OTHER_MODEL=mistralai/Mistral-Small-24B-Instruct-2501 sbatch ... --array=1-4 --export=ALL,XM_STAGE=other cluster/60_crossmodel.sbatch
#
# Outsiders use the same train.py defaults as M1 (4000 steps, LoRA r=8); extra flags in
# M2_TRAIN_ARGS, evaluation flags in XM_EVAL_ARGS.  Every stage is resumable.
# ---------------------------------------------------------------------------
#SBATCH --job-name=bi-xm
#SBATCH --account=torch_pr_1363_cds
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --constraint="h200"
#SBATCH --time=24:00:00
#SBATCH --output=logs/bi-xm-%x-%A_%a.out
#SBATCH --error=logs/bi-xm-%x-%A_%a.err
set -euo pipefail
source "${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}/cluster/job_common.sh"
STAGE="${XM_STAGE:?set XM_STAGE=record, same, other or summary}"
if [ "$STAGE" = summary ]; then
  in_container "python -m boolintro.crossmodel summarize --exps $RUNS_ROOT/*_L*_xm --out $RUNS_ROOT/xm_summary"
  exit 0
fi
resolve_levels
EXP="${RUN_DIR}_xm"
case "$STAGE" in
  record)
    CMD="python -m boolintro.crossmodel record --m1-run $RUN_DIR --exp $EXP" ;;
  same)
    SELF_MODEL=$(python3 -c "import json;print(json.load(open('$RUN_DIR/train_config.json'))['model'])")
    CMD="python -m boolintro.train --run-dir $EXP/m2_same --model $SELF_MODEL --init-seed ${SAME_INIT_SEED:-1} \
           --spec-from $EXP/spec.json --labels-from $EXP/m1_choices.json ${M2_TRAIN_ARGS:-}
         python -m boolintro.crossmodel evaluate --exp $EXP --m2-run $EXP/m2_same ${XM_EVAL_ARGS:-}" ;;
  other)
    : "${OTHER_MODEL:?set OTHER_MODEL, e.g. mistralai/Mistral-Small-24B-Instruct-2501}"
    CMD="python -m boolintro.train --run-dir $EXP/m2_other --model $OTHER_MODEL \
           --spec-from $EXP/spec.json --labels-from $EXP/m1_choices.json ${M2_TRAIN_ARGS:-}
         python -m boolintro.crossmodel evaluate --exp $EXP --m2-run $EXP/m2_other ${XM_EVAL_ARGS:-}" ;;
  *) echo "unknown XM_STAGE=$STAGE" >&2; exit 1 ;;
esac
nvidia-smi || true
in_container "$CMD"
