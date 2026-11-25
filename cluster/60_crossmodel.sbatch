#!/bin/bash
# ---------------------------------------------------------------------------
# 60 — Cross-model control ("Looking Inward"), one array task per level.
#
# Self model M1 = $RUNS_ROOT/${TAG}_L<level> (a finished 10_train run).  Two outsiders are
# trained on M1's recorded choices with the same recipe, then all three write self-reports:
#   same    a fresh LoRA on M1's own base model (MODEL of the M1 run), --init-seed $SAME_INIT_SEED
#   other   a different family: $OTHER_MODEL
#
#   EXTRA_MODELS=mistralai/Mistral-Small-24B-Instruct-2501 bash $REPO/cluster/02_prefetch.sh   # login node
#   sbatch --account="$TORCH_ACCOUNT" --chdir="$REPO" --array=1-4 \
#          --export=ALL,OTHER_MODEL=mistralai/Mistral-Small-24B-Instruct-2501 cluster/60_crossmodel.sbatch
#
# Role swap: train the other family as a self model too (10_train with MODEL=<other> TAG=<other tag>,
# same seed so the characters are identical), then rerun this script with TAG=<other tag> and
# OTHER_MODEL=$MODEL of the first run.  The summary over all ${XM_TAG}-tagged experiments then shows
# every pairing side by side.
#
# Results: $RUNS_ROOT/${TAG}_L<level>_xm/ and $RUNS_ROOT/xm_summary/report.md.
# Extra train.py flags for both outsiders in M2_TRAIN_ARGS; evaluation flags in XM_EVAL_ARGS.
# Every step is resumable: resubmit after preemption.  Set SKIP_SAME=1 or SKIP_OTHER=1 to drop one.
# ---------------------------------------------------------------------------
#SBATCH --job-name=bi-xm
#SBATCH --account=torch_pr_1363_cds
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --constraint="h200"
#SBATCH --time=48:00:00
#SBATCH --output=logs/bi-xm-%A_%a.out
#SBATCH --error=logs/bi-xm-%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kk6081@nyu.edu
set -euo pipefail
source "${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}/cluster/job_common.sh"
resolve_levels
EXP="${RUN_DIR}_xm"
SELF_MODEL=$(python3 -c "import json;print(json.load(open('$RUN_DIR/train_config.json'))['model'])")
SAME_INIT_SEED="${SAME_INIT_SEED:-1}"
STEPS=""
[ -z "${SKIP_SAME:-}" ] && STEPS+="
  python -m boolintro.train --run-dir $EXP/m2_same --model $SELF_MODEL --init-seed $SAME_INIT_SEED \
      --spec-from $EXP/spec.json --labels-from $EXP/m1_choices.json ${M2_TRAIN_ARGS:-}
  python -m boolintro.crossmodel evaluate --exp $EXP --m2-run $EXP/m2_same ${XM_EVAL_ARGS:-}"
if [ -z "${SKIP_OTHER:-}" ]; then
  : "${OTHER_MODEL:?set OTHER_MODEL (e.g. mistralai/Mistral-Small-24B-Instruct-2501) or SKIP_OTHER=1}"
  STEPS+="
  python -m boolintro.train --run-dir $EXP/m2_other --model $OTHER_MODEL \
      --spec-from $EXP/spec.json --labels-from $EXP/m1_choices.json ${M2_TRAIN_ARGS:-}
  python -m boolintro.crossmodel evaluate --exp $EXP --m2-run $EXP/m2_other ${XM_EVAL_ARGS:-}"
fi
nvidia-smi || true
in_container "
  python -m boolintro.crossmodel record --m1-run $RUN_DIR --exp $EXP
  $STEPS
  python -m boolintro.crossmodel summarize --exps $RUNS_ROOT/*_L*_xm --out $RUNS_ROOT/xm_summary
"
