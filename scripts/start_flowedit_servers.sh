#!/usr/bin/env bash
# Start 8 FlowEdit (Qwen-Image-Edit) vllm-omni servers, one per GPU.
# Each server binds to port 809{rank} and uses gpu_memory_utilization=0.30.
#
# Usage (inside Koala pod, after setup):
#   bash scripts/start_flowedit_servers.sh
#
# This script backgrounds all 8 servers and waits for health checks.
# The main training process should be launched AFTER this script completes.

set -euo pipefail

MODEL_PATH="${FLOWEDIT_MODEL_PATH:-/local-ssd/qwen-image-edit-2511}"
NUM_GPUS="${NUM_FLOWEDIT_GPUS:-8}"
GPU_MEM_UTIL="${FLOWEDIT_GPU_MEM_UTIL:-0.30}"
BASE_PORT=8090
HEALTH_TIMEOUT=180  # seconds to wait per server

if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    echo "Extract with: tar xf /local-ssd/qwen-image-edit-2511.tar -C /local-ssd/"
    exit 1
fi

echo "Starting $NUM_GPUS FlowEdit servers (model: $MODEL_PATH, mem: $GPU_MEM_UTIL)..."

PIDS=()
for rank in $(seq 0 $((NUM_GPUS - 1))); do
    port=$((BASE_PORT + rank))
    log="/tmp/flowedit_server_${rank}.log"

    CUDA_VISIBLE_DEVICES=$rank /tmp/vllm-venv/bin/python \
        -m vllm_omni.entrypoints.cli.main serve \
        "$MODEL_PATH" \
        --omni \
        --port "$port" \
        --host 0.0.0.0 \
        --model-class-name QwenImageFlowEditPipeline \
        --gpu-memory-utilization "$GPU_MEM_UTIL" \
        --max-model-len 8192 \
        --trust-remote-code \
        --dtype bfloat16 \
        > "$log" 2>&1 &

    PIDS+=($!)
    echo "  rank=$rank  port=$port  pid=${PIDS[-1]}  log=$log"
done

echo ""
echo "Waiting for servers to become healthy..."

all_healthy=true
for rank in $(seq 0 $((NUM_GPUS - 1))); do
    port=$((BASE_PORT + rank))
    url="http://localhost:${port}/health"
    start_time=$(date +%s)

    while true; do
        if curl -s -o /dev/null -w "%{http_code}" "$url" 2>/dev/null | grep -q "200"; then
            echo "  rank=$rank (port=$port): HEALTHY"
            break
        fi

        elapsed=$(( $(date +%s) - start_time ))
        if [ $elapsed -gt $HEALTH_TIMEOUT ]; then
            echo "  rank=$rank (port=$port): TIMEOUT after ${HEALTH_TIMEOUT}s"
            all_healthy=false
            break
        fi
        sleep 2
    done
done

echo ""
if [ "$all_healthy" = true ]; then
    echo "All $NUM_GPUS FlowEdit servers are healthy. PIDs: ${PIDS[*]}"
    echo "Ready to launch training."
else
    echo "WARNING: Some servers failed to start. Check /tmp/flowedit_server_*.log"
    exit 1
fi
