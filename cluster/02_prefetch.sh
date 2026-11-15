#!/bin/bash
# ---------------------------------------------------------------------------
# 02 — Download the models on the LOGIN node (compute nodes have no internet).
#
#   source /scratch/$USER/intros/cluster/env.sh
#   bash $REPO/cluster/02_prefetch.sh                  # MODEL and SMOKE_MODEL
#   EXTRA_MODELS="Qwen/Qwen3-8B Qwen/Qwen3-14B" bash $REPO/cluster/02_prefetch.sh
# ---------------------------------------------------------------------------
set -euo pipefail
REPO="${REPO:-/scratch/$USER/intros}"
source "$REPO/cluster/env.sh"

if [ ! -d /ext3 ] && [ "${BI_IN_CONTAINER:-0}" != "1" ]; then
  [ -f "$OVERLAY" ] || { echo "ERROR: overlay not found: $OVERLAY (see cluster/README.md)" >&2; exit 1; }
  export BI_IN_CONTAINER=1
  exec apptainer exec --overlay "$OVERLAY:ro" "$SIF" /bin/bash "$REPO/cluster/02_prefetch.sh"
fi
source /ext3/env.sh
for m in $MODEL $SMOKE_MODEL ${EXTRA_MODELS:-}; do
  echo "== downloading $m =="
  python - "$m" <<'PY'
import sys
from huggingface_hub import snapshot_download
print("->", snapshot_download(sys.argv[1], allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"]))
PY
done
echo "Prefetch done.  HF cache: $HF_HOME ($(du -sh "$HF_HOME" 2>/dev/null | cut -f1))"
