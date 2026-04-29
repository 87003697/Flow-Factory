#!/usr/bin/env bash
# scripts/data/score_ltx_syn_data.sh
# Multi-GPU launcher for offline LTX2 latent scoring.
#
# Environment variables (with defaults):
#   NPROC                  Number of GPUs to use (defaults to count of CUDA_VISIBLE_DEVICES, or `nvidia-smi -L | wc -l`).
#   REWARDS                Comma-separated reward names (subset of yaml). Default: "imagebind,clap".
#                          Add `,unified_reward_video_aps` only when the vLLM server is up.
#   OUT_DIR                Override yaml `data.out_dir`.
#   LIMIT                  Process at most N samples (passes through to --limit).
#   KEEP_DECODED           one of {all,first_n,none}. Default first_n.
#   KEEP_DECODED_FIRST_N   How many decoded mp4/wav to keep per rank when KEEP_DECODED=first_n. Default 4.
#   VAE_DTYPE              bfloat16|float16|float32. If colour artefacts appear, try float16/float32.
#   CONFIG                 YAML path. Default scripts/data/score_ltx_syn_data.yaml.
#   TORCHRUN_PORT          Master port for torchrun. Default 29501.
#
# Pre-flight (do this once per machine session):
#   1. Free GPUs: nvidia-smi; kill any stale training PIDs.
#   2. (Optional) Start UnifiedReward vLLM (e.g. on GPU 7):
#        CUDA_VISIBLE_DEVICES=7 bash scripts/data/start_unified_reward_server.sh
#      then verify: curl -s http://localhost:8080/v1/models | head
#   3. CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 NPROC=7 bash scripts/data/score_ltx_syn_data.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
cd "${REPO_ROOT}"

# --- Activate conda env (env name in this repo is `flow-factory` despite the dashed/underscore confusion)
if [[ -z "${CONDA_DEFAULT_ENV:-}" || "${CONDA_DEFAULT_ENV}" != "flow-factory" ]]; then
  if [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
    conda activate flow-factory
  fi
fi

CONFIG="${CONFIG:-scripts/data/score_ltx_syn_data.yaml}"
REWARDS="${REWARDS:-imagebind,clap}"
TORCHRUN_PORT="${TORCHRUN_PORT:-29501}"
KEEP_DECODED="${KEEP_DECODED:-first_n}"
KEEP_DECODED_FIRST_N="${KEEP_DECODED_FIRST_N:-4}"

# Resolve NPROC: prefer CUDA_VISIBLE_DEVICES width, fall back to nvidia-smi.
if [[ -z "${NPROC:-}" ]]; then
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    NPROC="$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')"
  else
    NPROC="$(nvidia-smi -L | wc -l)"
  fi
fi

extra_args=()
[[ -n "${LIMIT:-}" ]] && extra_args+=(--limit "${LIMIT}")
[[ -n "${OUT_DIR:-}" ]] && extra_args+=(--out_dir "${OUT_DIR}")
[[ -n "${VAE_DTYPE:-}" ]] && extra_args+=(--vae_dtype "${VAE_DTYPE}")

echo "[score_ltx_syn_data.sh] NPROC=${NPROC} REWARDS=${REWARDS} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "[score_ltx_syn_data.sh] CONFIG=${CONFIG} KEEP_DECODED=${KEEP_DECODED}"

torchrun \
  --standalone \
  --nproc_per_node="${NPROC}" \
  --master_port="${TORCHRUN_PORT}" \
  scripts/data/score_ltx_syn_data.py \
  --config "${CONFIG}" \
  --rewards "${REWARDS}" \
  --keep_decoded "${KEEP_DECODED}" \
  --keep_decoded_first_n "${KEEP_DECODED_FIRST_N}" \
  "${extra_args[@]}"
