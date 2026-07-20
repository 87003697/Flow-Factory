#!/usr/bin/env bash
# Launch contrastive FlowEdit OPD training on Koala normal pod.
# Orchestrates: setup → vllm-venv restore → FlowEdit servers → training.
#
# Usage (in koala submit -c):
#   cd /data/work/run_codes && [TRAIN_YAML=<path>] bash scripts/launch_contrastive_train.sh
#
# TRAIN_YAML env var overrides the default YAML (tex_contrastive.yaml).
# Note: `train.max_epochs=X` style CLI overrides are NOT supported — the
# training entrypoint (`ff-train`) discards unknown args. Use a distinct YAML.
#
# Example:
#   koala submit -m normal -g 8 --s3-log --code "$S3:/data/work/run_codes" \
#       -c "cd /data/work/run_codes && TRAIN_YAML=examples/opd/lora/trellis2/tex_contrastive_abl_any.yaml bash scripts/launch_contrastive_train.sh"
set -euo pipefail

PROJECT_DIR="/data/work/run_codes"
cd "$PROJECT_DIR"

# ============================================================================
# Phase 1: Core environment (training venv)
# ============================================================================
echo "=== Phase 1: Core environment setup ==="
. scripts/setup_koala.sh --fast

# ============================================================================
# Phase 2: vllm-omni venv (FlowEdit serving)
# ============================================================================
echo ""
echo "=== Phase 2: vllm-omni venv ==="
VLLM_VENV="/tmp/vllm-venv"
VLLM_TAR="s3://arcwm-code-us-west-2/ericzyma/data/flow_grpo/vllm_omni_venv.tar"

if [ -f "${VLLM_VENV}/bin/python" ]; then
    echo "  vllm-venv already present"
else
    echo "  Restoring vllm-venv from S3 (~9.7 GB, ~20s)..."
    s5cmd cat "${VLLM_TAR}" | tar xf - -C /tmp/
    echo "  Done"
fi

# Verify
if ! "${VLLM_VENV}/bin/python" -c "import vllm_omni" 2>/dev/null; then
    echo "  ERROR: vllm_omni import failed"
    exit 1
fi
echo "  vllm-omni OK (vllm $(${VLLM_VENV}/bin/python -c 'import vllm; print(vllm.__version__)'))"

# ============================================================================
# Phase 3: Start FlowEdit servers
# ============================================================================
echo ""
echo "=== Phase 3: FlowEdit servers ==="
bash scripts/start_flowedit_servers.sh

# ============================================================================
# Phase 4: Training
# ============================================================================
echo ""
echo "=== Phase 4: Training ==="
YAML="${TRAIN_YAML:-examples/opd/lora/trellis2/tex_contrastive.yaml}"

if [ ! -f "${YAML}" ]; then
    echo "ERROR: YAML not found: ${YAML}"
    exit 1
fi

echo "Config: ${YAML}"
echo ""

# ff-train is the CLI entry point (installed in /tmp/uv-venv/bin by uv sync)
exec ff-train "${YAML}"
