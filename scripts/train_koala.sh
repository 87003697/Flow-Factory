#!/bin/bash
# ============================================================================
# Flow-Factory (trellis2) — 训练入口脚本
# ============================================================================
#
# 功能：胶水脚本，把「环境恢复」和「启动训练」粘在一起。
#       设计为 koala submit -c 的目标命令，一行搞定 setup + train。
#
# 运行位置：Koala Pod 内（由 koala submit -c 自动调用）
#
# 用法：
#   bash scripts/train_koala.sh <config.yaml> [extra_args...]
#
# 示例：
#   # 单机 8 卡（ff-train 自动检测 GPU 数量）
#   bash scripts/train_koala.sh examples/grpo/lora/trellis2/shape.yaml
#
#   # 手动指定进程数（覆盖 yaml 中的 num_processes）
#   bash scripts/train_koala.sh examples/grpo/lora/trellis2/shape.yaml --num_processes 8
#
# 完整提交命令（在 Mac 上运行）：
#   bash scripts/koala_submit.sh -m normal -g 8 \
#       -c "cd /data/work/flow-factory && bash scripts/train_koala.sh examples/grpo/lora/trellis2/shape.yaml"
#
# 多机训练（Koala 自动注入 MASTER_ADDR/NUM_MACHINES/MACHINE_RANK 等环境变量，
#           ff-train 的 cli.py 会自动检测并传递给 accelerate launch）：
#   bash scripts/koala_submit.sh -m normal -n 2 -g 8 \
#       -c "cd /data/work/flow-factory && bash scripts/train_koala.sh examples/grpo/lora/trellis2/shape.yaml"
# ============================================================================
set -euo pipefail

CONFIG=${1:?"Usage: bash scripts/train_koala.sh <config.yaml> [extra_args...]"}
shift  # 移除第一个参数，$@ 变为 extra_args

# --- 日志持久化 ---
# normal 模式 Pod 崩溃后 koala logs 就没了。
# 后台每 30s 把日志同步到 S3（即使 OOM SIGKILL 也能拿到最近一次快照）。
LOG_FILE="/tmp/ff_train.log"
S3_LOG="s3://arcwm-code-us-west-2/${KOALA_USER:-ericzyma}/experiments/flow-factory-trellis2/train.log"

# 后台日志上传器（每 5s 上传一次，确保 crash 前最后几行能被捕获）
(while true; do sleep 5; aws s3 cp "${LOG_FILE}" "${S3_LOG}" --quiet 2>/dev/null || true; done) &
LOG_UPLOADER_PID=$!

# 清理函数：最后一次上传 + 杀后台进程
cleanup_log() {
    aws s3 cp "${LOG_FILE}" "${S3_LOG}" --quiet 2>/dev/null || true
    kill ${LOG_UPLOADER_PID} 2>/dev/null || true
}
trap cleanup_log EXIT

# 初始化环境（不加 --fast 会启动后台 S3 sync，正式训练需要）
# ⚠️ 必须直接 source，不能管道 (| tee)，否则 export 丢失
. scripts/setup_koala.sh

echo "" | tee -a "${LOG_FILE}"
echo "=== Starting training ===" | tee -a "${LOG_FILE}"
echo "Config: ${CONFIG}" | tee -a "${LOG_FILE}"
[ $# -gt 0 ] && echo "Extra args: $@" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

# Debug: GPU 显存状态
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv 2>&1 | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

# Debug: 打印 Koala 注入的分布式相关环境变量（精简输出）
echo "=== Distributed env vars ===" | tee -a "${LOG_FILE}"
(env | grep -E "^(RANK|LOCAL_RANK|WORLD_SIZE|MASTER_ADDR|MASTER_PORT|NUM_MACHINES|MACHINE_RANK|NUM_NODES|NODE_RANK)=" || echo "(none)") 2>&1 | tee -a "${LOG_FILE}"
echo "=============================" | tee -a "${LOG_FILE}"

# Koala 调度器会注入 RANK 等变量（节点标记），但并未用 torchrun 启动进程。
# ff-train 的 cli.py 看到 RANK 存在会误判为"已在分布式 launcher 内"，跳过 accelerate launch。
# 清除这些变量，让 ff-train 正确走 accelerate launch 多 GPU 路径。
unset RANK LOCAL_RANK WORLD_SIZE

# ff-train 是 Flow-Factory 的 CLI 入口（src/flow_factory/cli.py）：
# 1. 解析 YAML 配置
# 2. 自动检测多机环境变量（MASTER_ADDR, NUM_MACHINES 等 — Koala 多机模式注入）
# 3. 推断 num_processes（如 YAML 未指定，则 = num_machines × gpu_count）
# 4. 构建 accelerate launch 命令并启动训练
# stdbuf -oL: 行缓冲，确保每行立即写入文件（不等 buffer 满才 flush）
# PYTHONUNBUFFERED=1: 禁用 Python 输出缓冲（子进程 stderr 也能及时捕获）
PYTHONUNBUFFERED=1 stdbuf -oL ff-train "${CONFIG}" "$@" 2>&1 | tee -a "${LOG_FILE}"
