#!/usr/bin/env bash
# scripts/eval/run_inference.sh
#
# Run Flow-Factory inference for a given model config and checkpoint.
#
# Usage:
#   bash scripts/eval/run_inference.sh <config> [options]
#
# Examples:
#   # Basic – use prompts defined in the YAML
#   bash scripts/eval/run_inference.sh examples/inference/lora/flux1.yaml
#
#   # Override checkpoint and output dir
#   bash scripts/eval/run_inference.sh examples/inference/lora/flux1.yaml \
#       --checkpoint_path saves/flux1_lora_grpo_20260331_122628/checkpoints/checkpoint-1980 \
#       --output_dir outputs/my_eval
#
#   # Quick ad-hoc test with inline prompts
#   bash scripts/eval/run_inference.sh examples/inference/lora/flux1.yaml \
#       --checkpoint_path saves/flux1_lora_grpo_20260331_122628/checkpoints/checkpoint-1980 \
#       --prompts "A futuristic cyberpunk city at night" "A white cat on a windowsill"
#
# Environment:
#   CONDA_ENV   conda environment to use (default: flow-factory)

set -euo pipefail

CONDA_ENV="${CONDA_ENV:-flow-factory}"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <config.yaml> [ff-inference options...]"
    exit 1
fi

CONFIG="$1"
shift   # remaining args forwarded verbatim to ff-inference

cd "$(dirname "$0")/../.."   # repo root

echo "========================================"
echo "Flow-Factory Inference"
echo "  Config : $CONFIG"
echo "  Env    : $CONDA_ENV"
echo "  Args   : $*"
echo "========================================"

conda run -n "$CONDA_ENV" python -m flow_factory.inference "$CONFIG" "$@"
