#!/usr/bin/env bash
# scripts/eval/test_checkpoints.sh
#
# Run inference across multiple checkpoints of a training run to track training progress.
#
# Usage:
#   bash scripts/eval/test_checkpoints.sh <run_dir> [options]
#
# Arguments:
#   run_dir          Path to the training run directory
#                    (e.g. saves/flux1_lora_grpo_20260331_122629)
#
# Options:
#   -c, --config     Inference YAML config (default: examples/inference/lora/flux1.yaml)
#   -s, --step       Test every N-th checkpoint in sorted order (default: 1 = all)
#   -o, --output     Base output dir (default: outputs/<run_name>)
#   --ckpts          Space-separated specific step numbers to test, e.g. "80 500 1980"
#
# Examples:
#   # Test every checkpoint
#   bash scripts/eval/test_checkpoints.sh saves/flux1_lora_grpo_20260331_122629
#
#   # Test every 5th checkpoint (80, 180, 280, …) with a custom config
#   bash scripts/eval/test_checkpoints.sh saves/flux1_lora_grpo_20260331_122629 \
#       --step 5 --config examples/inference/lora/flux1.yaml
#
#   # Test hand-picked steps only
#   bash scripts/eval/test_checkpoints.sh saves/flux1_lora_grpo_20260331_122629 \
#       --ckpts "80 500 1000 1980"
#
# Environment:
#   CONDA_ENV   conda environment to use (default: flow-factory)

set -euo pipefail

CONDA_ENV="${CONDA_ENV:-flow-factory}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #
CONFIG="examples/inference/lora/flux1.yaml"
STEP_EVERY=1
OUTPUT_BASE=""
SPECIFIC_CKPTS=""
RUN_DIR=""

# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <run_dir> [--config YAML] [--step N] [--output DIR] [--ckpts '80 500 1980']"
    exit 1
fi

RUN_DIR="$1"
shift

while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--config)  CONFIG="$2";       shift 2 ;;
        -s|--step)    STEP_EVERY="$2";   shift 2 ;;
        -o|--output)  OUTPUT_BASE="$2";  shift 2 ;;
        --ckpts)      SPECIFIC_CKPTS="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

cd "$REPO_ROOT"

# Resolve run dir (allow relative to repo root)
if [[ ! -d "$RUN_DIR" ]]; then
    echo "Error: run directory not found: $RUN_DIR"
    exit 1
fi

RUN_NAME="$(basename "$RUN_DIR")"
CKPT_BASE="$RUN_DIR/checkpoints"

if [[ -z "$OUTPUT_BASE" ]]; then
    OUTPUT_BASE="outputs/${RUN_NAME}"
fi

# --------------------------------------------------------------------------- #
# Collect checkpoints to test
# --------------------------------------------------------------------------- #
if [[ -n "$SPECIFIC_CKPTS" ]]; then
    # User provided explicit steps
    STEPS_TO_TEST=($SPECIFIC_CKPTS)
else
    # Discover all checkpoint-* dirs, sort numerically
    mapfile -t ALL_STEPS < <(
        find "$CKPT_BASE" -maxdepth 1 -type d -name "checkpoint-*" \
            | sed 's|.*checkpoint-||' | sort -n
    )
    if [[ ${#ALL_STEPS[@]} -eq 0 ]]; then
        echo "Error: no checkpoints found in $CKPT_BASE"
        exit 1
    fi
    # Apply --step stride
    STEPS_TO_TEST=()
    for i in "${!ALL_STEPS[@]}"; do
        if (( i % STEP_EVERY == 0 )); then
            STEPS_TO_TEST+=("${ALL_STEPS[$i]}")
        fi
    done
fi

# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
echo "========================================"
echo "Flow-Factory  ·  Checkpoint Sweep"
echo "  Run      : $RUN_DIR"
echo "  Config   : $CONFIG"
echo "  Checkpts : ${#STEPS_TO_TEST[@]}  (step-every $STEP_EVERY)"
echo "  Output   : $OUTPUT_BASE"
echo "  Env      : $CONDA_ENV"
echo "========================================"

FAILED=()

for STEP in "${STEPS_TO_TEST[@]}"; do
    CKPT_PATH="${CKPT_BASE}/checkpoint-${STEP}"
    OUT_DIR="${OUTPUT_BASE}/checkpoint-${STEP}"

    if [[ ! -d "$CKPT_PATH" ]]; then
        echo "[SKIP] checkpoint-${STEP} not found at $CKPT_PATH"
        continue
    fi

    echo ""
    echo "----------------------------------------"
    echo "  checkpoint-${STEP}  →  $OUT_DIR"
    echo "----------------------------------------"

    if conda run -n "$CONDA_ENV" python -m flow_factory.inference "$CONFIG" \
            --checkpoint_path "$CKPT_PATH" \
            --output_dir "$OUT_DIR"; then
        echo "[OK] checkpoint-${STEP}"
    else
        echo "[FAILED] checkpoint-${STEP}"
        FAILED+=("$STEP")
    fi
done

# --------------------------------------------------------------------------- #
# Final report
# --------------------------------------------------------------------------- #
echo ""
echo "========================================"
echo "Sweep complete."
echo "  Tested  : ${#STEPS_TO_TEST[@]} checkpoints"
echo "  Output  : $OUTPUT_BASE"
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "  FAILED  : ${FAILED[*]}"
    exit 1
else
    echo "  All checkpoints passed."
fi
echo "========================================"
