#!/bin/bash
# ---------------------------------------------------------------------------
# Submit the full experiment as a dependency chain (run on the login node):
#   train (array 1-4)  ->  eval (array 1-4, task i waits for train task i)  ->  analyze
#
#   source /scratch/$USER/intros/cluster/env.sh
#   bash $REPO/cluster/submit_all.sh
#   MODEL=Qwen/Qwen3-8B TAG=qwen3-8b ARRAY=1-4 bash $REPO/cluster/submit_all.sh
#   LEVELS=1,2,3,4 ARRAY= TAG=qwen3-32b-mixed bash $REPO/cluster/submit_all.sh   # one mixed run
# ---------------------------------------------------------------------------
set -euo pipefail
REPO="${REPO:-/scratch/$USER/intros}"
source "$REPO/cluster/env.sh"
cd "$REPO"
mkdir -p logs
ARRAY="${ARRAY-1-4}"
ARR=(); [ -n "$ARRAY" ] && ARR=(--array="$ARRAY")
COMMON=(--parsable --account="$TORCH_ACCOUNT" --chdir="$REPO" --export=ALL)

TRAIN=$(sbatch "${COMMON[@]}" "${ARR[@]}" cluster/10_train.sbatch)
DEP="afterok:$TRAIN"; [ -n "$ARRAY" ] && DEP="aftercorr:$TRAIN"
EVAL=$(sbatch "${COMMON[@]}" "${ARR[@]}" --dependency="$DEP" cluster/20_eval.sbatch)
ANALYZE=$(sbatch "${COMMON[@]}" --dependency="afterany:$EVAL" cluster/30_analyze.sbatch)
echo "train=$TRAIN eval=$EVAL analyze=$ANALYZE  (tag=$TAG model=$MODEL)"
echo "watch with: squeue --me ; results in $RUNS_ROOT/${TAG}_analysis"
