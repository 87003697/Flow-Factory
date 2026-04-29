#!/usr/bin/env bash
# scripts/data/start_unified_reward_server.sh
# Start a vLLM OpenAI-compatible server hosting CodeGoat24/UnifiedReward-2.0
# (or any compatible UnifiedReward weight) for use by the
# `unified_reward_video_aps` reward in `scripts/data/score_ltx_syn_data.py`.
#
# Recommended single-card placement: keep this off the GPUs used for decoding.
#   CUDA_VISIBLE_DEVICES=7 bash scripts/data/start_unified_reward_server.sh
# then from another shell:
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 NPROC=7 bash scripts/data/score_ltx_syn_data.sh
#
# Verify after startup:
#   curl -s http://localhost:8080/v1/models | head
#
# Optional environment variables:
#   MODEL_PATH                Default: CodeGoat24/UnifiedReward-2.0-qwen35-9b
#   SERVED_MODEL_NAME         OpenAI "model" id; must match yaml `vlm_model`. Default: UnifiedReward
#   PORT                      Default: 8080
#   HOST                      Default: 0.0.0.0
#   TENSOR_PARALLEL_SIZE      Default: 1
#   DATA_PARALLEL_SIZE        Default: count of CUDA_VISIBLE_DEVICES, else 1
#   GPU_MEMORY_UTILIZATION    Default: 0.9
#   LIMIT_MM_PER_PROMPT_IMAGE Default: 20  (matches max_frames + a few condition images)
#   VLLM_BIN                  Default: vllm
#   Any extra args after `--` are forwarded to `vllm serve`.
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-CodeGoat24/UnifiedReward-2.0-qwen35-9b}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-UnifiedReward}"
PORT="${PORT:-8080}"
HOST="${HOST:-0.0.0.0}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
LIMIT_MM_PER_PROMPT_IMAGE="${LIMIT_MM_PER_PROMPT_IMAGE:-20}"
VLLM_BIN="${VLLM_BIN:-vllm}"

case "${DATA_PARALLEL_SIZE-unset}" in
  unset)
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
      DATA_PARALLEL_SIZE="$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')"
    else
      DATA_PARALLEL_SIZE=1
    fi
    ;;
esac

if [[ -z "${CONDA_DEFAULT_ENV:-}" || "${CONDA_DEFAULT_ENV}" != "vllm-serve" && "${CONDA_DEFAULT_ENV}" != "flow-factory" ]]; then
  if [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
    # Prefer dedicated vllm-serve env if present, else fall back to flow-factory.
    if conda env list | awk '{print $1}' | grep -qx vllm-serve; then
      conda activate vllm-serve
    else
      conda activate flow-factory
    fi
  fi
fi

echo "[start_unified_reward_server.sh] MODEL_PATH=${MODEL_PATH} SERVED_MODEL_NAME=${SERVED_MODEL_NAME}"
echo "[start_unified_reward_server.sh] HOST=${HOST} PORT=${PORT} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "[start_unified_reward_server.sh] DATA_PARALLEL_SIZE=${DATA_PARALLEL_SIZE} TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE}"

exec "${VLLM_BIN}" serve "${MODEL_PATH}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --data-parallel-size "${DATA_PARALLEL_SIZE}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --limit-mm-per-prompt "image=${LIMIT_MM_PER_PROMPT_IMAGE}" \
  "$@"
