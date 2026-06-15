#!/bin/bash
# Sentinel: 如果存在 /tmp/koala_submit_disabled，立即退出（用于挡住后台 retry loop）
if [ -f /tmp/koala_submit_disabled ]; then
    echo "❌ koala_submit disabled by sentinel /tmp/koala_submit_disabled. Remove that file to re-enable."
    exit 1
fi
# ============================================================================
# Flow-Factory (trellis2) — Mac 端一键提交到 Koala
# ============================================================================
#
# 功能：在 Mac 本地运行，完成两件事：
#       1. 把项目代码增量同步到 S3（排除大文件）
#       2. 调用 koala submit 提交任务到集群
#
# 运行位置：Mac 本地（~/Desktop/codes/Flow-Factory/）
#
# 用法：
#   bash scripts/koala_submit.sh                    # debug pod (1 GPU, 可 SSH, 12h)
#   bash scripts/koala_submit.sh -g 8              # debug pod (8 GPU, 可 SSH)
#   bash scripts/koala_submit.sh -m normal -g 8    # 正式训练 (8 GPU, 无 SSH, 48h)
#   bash scripts/koala_submit.sh -m normal -g 8 \
#       -c "cd /data/work/flow-factory && bash scripts/train_koala.sh examples/grpo/lora/trellis2/shape.yaml"
#
# 什么会被上传到 S3：
#   - 项目源码（src/, scripts/, examples/, config/, dataset/ 等）
#   - diffusers/ submodule（~50 MB，含自定义修改）
#
# 什么不会上传（太大，由 setup_koala.sh 从 S3 tar 恢复）：
#   - pretrained_weights/ (16+ GB 模型权重)
#   - third_party/ (TRELLIS.2 源码，361 MB tar)
#   - _reference_codes/ (如果有)
#   - .git/, .venv/, __pycache__/, wandb/, saves/ (无关文件)
#
# 前置条件：
#   - 已安装 aws CLI 且配置了 S3 凭证
#   - 已安装 koala CLI
#   - git submodule 已初始化（脚本会自动检查）
# ============================================================================
set -euo pipefail

# S3 上的代码存放路径（每次 sync 覆盖，多次提交复用同一 prefix）
PROJECT="flow-factory-trellis2"
S3="s3://arcwm-code-us-west-2/ericzyma/${PROJECT}"

# --- 确保 diffusers submodule 已初始化 ---
# diffusers 是 git submodule，clone 后默认为空目录，需要 init
if [ ! -f diffusers/setup.py ]; then
    echo "Initializing diffusers submodule..."
    git submodule update --init --recursive
fi

# --- 增量同步代码到 S3 ---
# aws s3 sync: 只上传有变化的文件（按 size + mtime 判断）
# --delete: 删除远端多余文件（防止旧文件残留，如 training_args.py→package 重构）
# --exclude: 排除不需要上传的大文件和临时文件
# --quiet: 不逐文件打印（减少输出噪音）
echo "Syncing code to S3: ${S3}/"
aws s3 sync . "${S3}/" \
    --delete \
    --exclude '.git/*' \
    --exclude '.venv/*' \
    --exclude '*/__pycache__/*' \
    --exclude '*.pyc' \
    --exclude 'wandb/*' \
    --exclude 'saves/*' \
    --exclude '.scratch/*' \
    --exclude 'pretrained_weights/*' \
    --exclude 'third_party/*' \
    --exclude '_reference_codes/*' \
    --quiet

echo "Code synced."
echo ""

# --- 提交到 Koala ---
# --code "$S3:/data/work/run_codes": 告诉 Pod 从 S3 拉代码到容器内指定路径
# "$@": 透传所有额外参数给 koala submit（如 -m normal, -g 8, -c "..." 等）
# LC_ALL / PYTHONIOENCODING: 防止含中文文件名时 koala CLI 报编码错误
#
# 注意：Koala 只自动注入 AWS 凭证，HF_TOKEN 和 WANDB_API_KEY 需要手动传递。
echo "Submitting to Koala..."

# 构建环境变量注入前缀（仅注入本地有值的变量）
ENV_INJECT=""
[ -n "${WANDB_API_KEY:-}" ] && ENV_INJECT="${ENV_INJECT}export WANDB_API_KEY='${WANDB_API_KEY}'; "
[ -n "${HF_TOKEN:-}" ] && ENV_INJECT="${ENV_INJECT}export HF_TOKEN='${HF_TOKEN}'; "

# 检查用户是否提供了 -c 参数，如果有则注入 env vars
ARGS=("$@")
for i in "${!ARGS[@]}"; do
    if [[ "${ARGS[$i]}" == "-c" || "${ARGS[$i]}" == "--cmd" ]]; then
        next=$((i + 1))
        if [ $next -lt ${#ARGS[@]} ]; then
            ARGS[$next]="${ENV_INJECT}${ARGS[$next]}"
        fi
        break
    fi
done

LC_ALL=en_US.UTF-8 PYTHONIOENCODING=utf-8 \
    koala submit --code "${S3}:/data/work/run_codes" "${ARGS[@]}"
