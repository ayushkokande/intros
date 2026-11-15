#!/bin/bash
# Run one command inside the container (login-node helper for light work).
#   bash cluster/run.sh python -m boolintro.preview
#   bash cluster/run.sh python -m pytest -q tests
set -euo pipefail
REPO="${REPO:-/scratch/$USER/intros}"
source "$REPO/cluster/env.sh"
[ "$#" -gt 0 ] || { echo "usage: run.sh <command> [args...]" >&2; exit 2; }
if [ -d /ext3 ]; then source /ext3/env.sh; cd "$REPO"; exec "$@"; fi
NV=""; [ -e /dev/nvidia0 ] && NV="--nv"
exec apptainer exec $NV --overlay "$OVERLAY:ro" "$SIF" /bin/bash -c '
  source /ext3/env.sh
  export HF_HOME="'"$HF_HOME"'" TORCH_HOME="'"$TORCH_HOME"'" TMPDIR="'"$TMPDIR"'"
  cd "'"$REPO"'"
  exec "$@"' _ "$@"
