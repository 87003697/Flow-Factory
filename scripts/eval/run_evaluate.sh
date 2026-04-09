#!/usr/bin/env bash
# scripts/eval/run_evaluate.sh
#
# Run Flow-Factory evaluation for a given metrics config.
#
# Usage:
#   bash scripts/eval/run_evaluate.sh <config> [options]
#
# Examples:
#   # Basic – use paths defined in the YAML
#   bash scripts/eval/run_evaluate.sh examples/eval/compute_metrics.yaml
#
#   # Override samples directory and output path
#   bash scripts/eval/run_evaluate.sh examples/eval/compute_metrics.yaml \
#       --samples_dir outputs/my_eval \
#       --output_path results/my_metrics.json
#
#   # List available metrics
#   bash scripts/eval/run_evaluate.sh examples/eval/compute_metrics.yaml --list_metrics
#
# Environment:
#   CONDA_ENV   conda environment to use (default: flow-factory)

set -euo pipefail

CONDA_ENV="${CONDA_ENV:-flow-factory}"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <config.yaml> [ff-evaluate options...]"
    exit 1
fi

CONFIG="$1"
shift   # remaining args forwarded verbatim to ff-evaluate

cd "$(dirname "$0")/../.."   # repo root

echo "========================================"
echo "Flow-Factory Evaluation"
echo "  Config : $CONFIG"
echo "  Env    : $CONDA_ENV"
echo "  Args   : $*"
echo "========================================"

conda run -n "$CONDA_ENV" python -m flow_factory.evaluate "$CONFIG" "$@"
