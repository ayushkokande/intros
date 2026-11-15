#!/bin/bash
# ---------------------------------------------------------------------------
# 01 — OPTIONAL.  Build a separate overlay environment for this project.
# Skip this if the probe_conf overlay exists (the default in env.sh).
#
#   source /scratch/$USER/intros/cluster/env.sh
#   export OVERLAY=/scratch/$USER/intros_overlay-15GB-500K.ext3   # and add to env.sh
#   srun --account="$TORCH_ACCOUNT" --cpus-per-task=4 --mem=16GB --time=02:00:00 --pty /bin/bash
#   cp -rp "$OVERLAY_SRC" "$OVERLAY.gz" && gunzip -f "$OVERLAY.gz"
#   apptainer exec --fakeroot --overlay "$OVERLAY:rw" "$SIF" /bin/bash
#   bash /scratch/$USER/intros/cluster/01_setup_overlay.sh
# ---------------------------------------------------------------------------
set -euo pipefail
if [ ! -d /ext3 ]; then
  echo "ERROR: /ext3 is not mounted; start the container with --overlay \$OVERLAY:rw" >&2
  exit 1
fi
REPO="${REPO:-/scratch/$USER/intros}"
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="/ext3/.uv" sh
source /ext3/.uv/env
export UV_PYTHON_INSTALL_DIR="/ext3/.uv/python" UV_CACHE_DIR="/ext3/.uv/cache"
uv venv /ext3/.venv --python 3.12
uv pip install --python /ext3/.venv/bin/python -r "$REPO/cluster/requirements.txt"
cat > /ext3/env.sh <<'EOS'
#!/bin/bash
unset -f which
source /ext3/.uv/env
source /ext3/.venv/bin/activate
export PATH=/ext3/.venv/bin:$PATH
EOS
chmod +x /ext3/env.sh
source /ext3/env.sh
python -c "import torch, transformers, safetensors, pandas; print('torch', torch.__version__, 'transformers', transformers.__version__)"
