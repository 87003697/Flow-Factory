#!/bin/bash
# ============================================================================
# KOALA 环境恢复脚本 — Flow-Factory (trellis2)
# ============================================================================
#
# 功能：在 Koala 集群 Pod 内一键恢复 Trellis2 GRPO 训练所需的全部环境。
#       包括 Python 依赖、CUDA 扩展、模型权重、数据集、后台持久化。
#
# 用法：
#   . scripts/setup_koala.sh [--fast] [--download]
#
#   --fast      日常恢复模式。假设 S3 上已有所有 tar 缓存，从 tar 恢复即可。
#               跳过后台 S3 sync（适合 debug pod 手动调试）。
#   --download  首次使用模式。从 HuggingFace/GitHub 下载权重/数据，
#               编译 CUDA 扩展，然后打包到 S3 供后续复用。
#   （不加参数） 完整恢复 + 启动后台 S3 sync（正式训练用）。
#
# ⚠️ 重要：必须通过 source 执行（. scripts/setup_koala.sh），
#          否则 export 的环境变量不会传递给当前 shell。
#
# 环境变量要求：
#   HF_TOKEN       HuggingFace 认证 token。
#                  Trellis2 pipeline 加载时需要下载 BiRefNet (briaai/RMBG-2.0)，
#                  这是一个 gated repo，没有 token 会报 403。
#                  ⚠️ Koala 不自动注入此变量，需在 Mac 的 ~/.zshrc 中配置。
#   WANDB_API_KEY  WandB 认证（可选，不设则训练不上报 wandb）。
#
# S3 资产布局（全部复用 flow_grpo_custom 项目已打好的 tar，无需重新打包）：
#   s3://arcwm-code-us-west-2/ericzyma/data/flow_grpo/
#     TRELLIS.2-4B.tar           (16 GB)  Trellis2 Shape+Tex Flow Model 权重
#     TRELLIS-image-large.tar    (3.1 GB) 原版 TRELLIS v1 共享组件（ss_dec decoder 等）
#     dinov3-vitl16.tar          (1.2 GB) DINOv3 图像编码器（Trellis2 条件输入）
#     qwen-image-edit-2511.tar   (39 GB)  FlowEdit Guidance 模型（UnifiedReward 用，可选）
#     alphaimages_v3.tar         (474 MB) 训练数据集（2396 张 RGBA 图片）
#     trellis2_reference.tar     (361 MB) TRELLIS.2 源码（含 trellis2 Python 包 + o-voxel）
#     cuda_site_packages.tar     (145 MB) 预编译 CUDA 扩展（nvdiffrast/CuMesh/FlexGEMM 的 .so）
#
# v1.4.0 适配：不再依赖 /threed-code FUSE 挂载。所有 tar 通过 s5cmd cat s3://... 直接
# 从 S3 API 拉取并管道解压到本地盘，更快更稳（无 FUSE 小文件性能问题）。
#
# 恢复后的目录结构：
#   /data/work/flow-factory/              ← 项目代码（koala --code 拉取）
#     ├── pretrained_weights → /local-ssd/pretrained_weights  (symlink)
#     ├── dataset/alphaimages_v3 → /local-ssd/alphaimages_v3  (symlink)
#     ├── third_party/TRELLIS.2/          ← 从 tar 恢复的 trellis2 源码
#     └── diffusers/                      ← git submodule（随代码上传）
#   /tmp/uv-venv/                         ← Python 虚拟环境
#   /local-ssd/pretrained_weights/        ← 模型权重（NVMe 高速盘）
#   /local-ssd/alphaimages_v3/            ← 数据集
#   /local-ssd/hf_cache/                  ← HuggingFace 下载缓存
#
# 耗时（H200 Pod，--fast 模式）：
#   [1/7] Python deps:    ~15s（uv sync，有 uv cache 命中）
#   [2/7] torch 覆盖:     ~10s
#   [3/7] extra deps:     ~15s（已安装则跳过；flash-attn 有缓存 ~7s）
#   [4/7] CUDA ext:       ~3s（tar 解压）
#   [5/7] TRELLIS.2:      ~5s（tar 解压）
#   [6/7] 权重+数据:      ~40s（16GB + 1.2GB + 474MB tar 解压）
#   [7/7] S3 sync:        instant（--fast 跳过）
#   总计:                  ~90s
# ============================================================================
set -euo pipefail

# --- 参数解析 ---
FAST_MODE=false
DOWNLOAD_MODE=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --fast)     FAST_MODE=true; shift ;;
        --download) DOWNLOAD_MODE=true; shift ;;
        *)          echo "Unknown option: $1"; return 1 2>/dev/null || exit 1 ;;
    esac
done

# --- 路径配置 ---
# KOALA_USER: koala CLI 自动注入的用户名（对应 S3 bucket 内的目录名）
USER="${KOALA_USER:-ericzyma}"
# v1.4.0: 不再依赖 /threed-code FUSE。所有 S3 资源通过 s5cmd 走 API 拉取。
S3_BUCKET="s3://arcwm-code-us-west-2/${USER}"
# S3 API 路径（写入用）
S3_DATA="${S3_BUCKET}/data/flow_grpo"
# 项目代码根目录（koala launch 时 cd 到的目录，自动检测）
PROJECT_DIR="$(pwd)"

# S3 tar URI（直接用 s5cmd cat 拉取，不走 FUSE）
TRELLIS2_TAR="${S3_DATA}/TRELLIS.2-4B.tar"       # Trellis2 4B 模型权重
DINOV3_TAR="${S3_DATA}/dinov3-vitl16.tar"         # DINOv3 图像编码器
QWEN_TAR="${S3_DATA}/qwen-image-edit-2511.tar"    # Qwen Guidance（UnifiedReward 用）
DATASET_TAR="${S3_DATA}/alphaimages_v2.tar"        # 训练数据集（含 prompt caption）
REFERENCE_TAR="${S3_DATA}/trellis2_reference.tar"  # TRELLIS.2 源码包
CUDA_SP_TAR="${S3_DATA}/cuda_site_packages.tar"    # 预编译 CUDA 扩展
TRELLIS1_TAR="${S3_DATA}/TRELLIS-image-large.tar"  # 原版 TRELLIS v1 共享组件（ss_dec 等）

# 本地高速盘路径（Pod 生命周期内有效，重启后丢失）
WEIGHTS_LOCAL="/local-ssd/pretrained_weights"
DATASET_LOCAL="/local-ssd/alphaimages_v2_formatted"
VENV="/tmp/uv-venv"

# koala launch already cd's here; verify the directory is correct
if [ ! -f "pyproject.toml" ]; then
    echo "ERROR: PROJECT_DIR=${PROJECT_DIR} does not contain pyproject.toml"
    return 1 2>/dev/null || exit 1
fi

# --- 环境变量 ---
# 把 venv 的 bin 加到 PATH 最前面，确保 python/ninja/ff-train 等命令可用
export PATH="${VENV}/bin:${PATH}"
# 告诉 uv 把包装到这个 venv（而非项目内 .venv）
export UV_PROJECT_ENVIRONMENT="${VENV}"
# HuggingFace 下载缓存指向本地高速盘（v1.4.0 起 /threed-code 默认不挂载）
export HF_HOME="/local-ssd/hf_cache"
# 禁用 HF 的 xet 下载器（在 S3 FUSE 路径写入时会 panic）
export HF_HUB_DISABLE_XET=1
export HF_TOKEN="${HF_TOKEN:-}"
# CUDA 编译目标架构（A100=8.0, A10=8.6, L40=8.9, H100/H200=9.0）
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
# PyTorch 内存分配优化：允许 reserved memory 的碎片化段被扩展复用，减少 OOM
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Trellis2 sparse transformer 的注意力后端选择
export ATTN_BACKEND=flash_attn
export WANDB_API_KEY="${WANDB_API_KEY:-}"
# TRELLIS.2 源码加入 Python 搜索路径
# trellis2.py 中有 sys.path.insert(0, "third_party/TRELLIS.2") 做同样的事，
# 但 PYTHONPATH 确保在任何 working directory 下都能 import trellis2
export PYTHONPATH="${PROJECT_DIR}/third_party/TRELLIS.2:${PYTHONPATH:-}"

# ============================================================================
# [1/7] Python venv + Flow-Factory 核心依赖
# ============================================================================
# 创建 venv 并安装 flow-factory 自身（pyproject.toml 定义的依赖）。
# ⚠️ uv sync 会根据 pyproject.toml 中 torch>=2.6.0 拉取最新 torch（如 2.12），
#    这会在 [2/7] 中被覆盖为 2.6.0+cu124。必须先 sync 再覆盖，不能反过来。
echo "=== [1/7] Python dependencies ==="
if [ ! -d "${VENV}" ]; then
    uv venv --python 3.12 "${VENV}"
fi

# 安装 flow-factory 包及其核心依赖（transformers, accelerate, deepspeed, wandb 等）
# --frozen: 不更新 uv.lock（避免触发 geneval/mmcv 的 resolve，那个在 Python 3.12 上构建失败）
if ! "${VENV}/bin/python" -c "import flow_factory" 2>/dev/null; then
    echo "  uv sync (flow-factory + deepspeed + wandb)..."
    uv sync --extra deepspeed --extra wandb --frozen 2>&1 | tail -3
else
    echo "  flow-factory already installed"
fi

# diffusers 使用项目内的 git submodule 版本（含自定义修改），editable 安装
# --no-deps: 不重复安装 diffusers 的依赖（已由 uv sync 处理）
if ! "${VENV}/bin/python" -c "import diffusers; assert 'flow-factory' in diffusers.__file__" 2>/dev/null; then
    echo "  Installing diffusers (editable)..."
    uv pip install --python "${VENV}/bin/python" -e ./diffusers --no-deps 2>&1 | tail -2
else
    echo "  diffusers already installed (editable)"
fi
echo "  Done (core)"

# ============================================================================
# [2/7] PyTorch 2.6.0+cu124（覆盖 uv sync 拉的最新版本）
# ============================================================================
# 为什么要锁 torch 2.6.0+cu124：
#   1. 预编译的 CUDA 扩展（nvdiffrast/cumesh/flex_gemm）是用这个版本编译的，
#      torch C++ ABI 跨大版本不兼容，换版本会报 undefined symbol
#   2. flash-attn 2.7.3 的 wheel 也绑定 torch 2.6.0
#   3. 与 flow_grpo_custom 保持一致，确保复用其 tar 资产
echo "=== [2/7] PyTorch 2.6.0+cu124 ==="
TORCH_VER=$("${VENV}/bin/python" -c "import torch; print(torch.__version__)" 2>/dev/null || echo "none")
if [[ "${TORCH_VER}" != 2.6.0* ]]; then
    echo "  Current: ${TORCH_VER}, installing 2.6.0+cu124..."
    uv pip install --python "${VENV}/bin/python" \
        torch==2.6.0+cu124 torchvision==0.21.0+cu124 \
        --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -3
else
    echo "  Already 2.6.0+cu124"
fi

# ============================================================================
# [3/7] Trellis2 额外依赖（不在 pyproject.toml 中）
# ============================================================================
# 这些包是 trellis2 源码（third_party/TRELLIS.2）和 o-voxel 运行时需要的，
# 但没有写进 Flow-Factory 的 pyproject.toml（因为只有 trellis2 分支需要）。
# - trimesh/plyfile/open3d: 3D 网格处理
# - kiui: 3D 视觉工具包
# - rembg: 背景移除（BiRefNet 推理）
# - utils3d: 渲染工具（fixed commit，API 不稳定所以锁版本）
# - ninja: torch JIT C++ 编译器调用（mesh_voxelize 等自定义 op）
# - kornia/timm: 图像变换和预训练 backbone
# - zstandard: o-voxel 的隐含依赖（未在其 setup.py 中声明）
echo "=== [3/7] Trellis2 extra dependencies ==="
if "${VENV}/bin/python" -c "import trimesh, utils3d, plyfile, kiui" 2>/dev/null; then
    echo "  Already installed"
else
    echo "  Installing trellis2 deps..."
    uv pip install --python "${VENV}/bin/python" \
        trimesh plyfile open3d kiui rembg onnxruntime \
        imageio-ffmpeg easydict opencv-python-headless ninja \
        kornia timm lpips pytorch-msssim zstandard \
        "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8" \
        2>&1 | tail -3
    echo "  Done"
fi

# flash-attn: Trellis2 sparse transformer 的高效注意力实现
# --no-build-isolation: flash-attn 的 setup.py 需要先有 wheel/setuptools/torch
# 首次编译 ~3min（从源码），后续有 uv wheel cache 命中只需 ~7s
if ! "${VENV}/bin/python" -c "import flash_attn" 2>/dev/null; then
    echo "  Compiling flash-attn (~3 min or cache hit ~5s)..."
    uv pip install --python "${VENV}/bin/python" wheel setuptools 2>/dev/null
    uv pip install --python "${VENV}/bin/python" \
        --no-build-isolation flash-attn==2.7.3 2>&1 | tail -3
else
    echo "  flash-attn already installed"
fi

# ============================================================================
# [4/7] CUDA 扩展（预编译 site-packages 恢复）
# ============================================================================
# nvdiffrast: NVIDIA 可微分渲染器（Trellis2 多视角渲染 reward 用）
# cumesh: CUDA 加速网格操作
# flex_gemm: Trellis2 sparse transformer 的高效矩阵乘法
#
# 这些扩展含 .so 文件，编译时绑定 torch ABI。
# cuda_site_packages.tar 是用 torch 2.6.0+cu124 + Python 3.12 编译好的，
# 直接解压到 site-packages/ 即可（~3s），无需重新编译（~8min）。
echo "=== [4/7] CUDA extensions ==="
if "${VENV}/bin/python" -c "import nvdiffrast, cumesh, flex_gemm" 2>/dev/null; then
    echo "  Already installed"
elif s5cmd ls "${CUDA_SP_TAR}" &>/dev/null; then
    echo "  Restoring pre-built packages from S3 tar..."
    s5cmd cat "${CUDA_SP_TAR}" | tar xf - -C "${VENV}/lib/python3.12/site-packages/"
    echo "  Restored (~3s)"
elif [ "$DOWNLOAD_MODE" = true ]; then
    echo "  ERROR: --download mode for CUDA ext not implemented yet."
    echo "  Please run flow_grpo_custom's setup first to create the tar."
    return 1 2>/dev/null || exit 1
else
    echo "  ERROR: No CUDA ext tar at ${CUDA_SP_TAR}."
    echo "  Ensure flow_grpo_custom's S3 tars exist."
    return 1 2>/dev/null || exit 1
fi

# ============================================================================
# [5/7] third_party/TRELLIS.2（代码 + o-voxel）
# ============================================================================
# TRELLIS.2 不是标准 pip 包，而是通过 sys.path / PYTHONPATH 导入的源码。
# 包含：trellis2/ Python 包（模型定义、稀疏 tensor、渲染器）+ o-voxel/（体素化）
# 来源：https://github.com/87003697/TRELLIS.2.git
#
# o-voxel 通过 pip install 安装（有 setup.py），但 --no-deps 因为依赖由 [3/7] 处理。
echo "=== [5/7] TRELLIS.2 reference code ==="
if [ -d "${PROJECT_DIR}/third_party/TRELLIS.2/trellis2" ]; then
    echo "  Already present"
elif s5cmd ls "${REFERENCE_TAR}" &>/dev/null; then
    echo "  Restoring from S3 tar..."
    mkdir -p "${PROJECT_DIR}/third_party"
    s5cmd cat "${REFERENCE_TAR}" | tar xf - -C "${PROJECT_DIR}/third_party/"
    echo "  Restored"
elif [ "$DOWNLOAD_MODE" = true ]; then
    echo "  Cloning TRELLIS.2..."
    mkdir -p "${PROJECT_DIR}/third_party"
    git clone --recursive https://github.com/87003697/TRELLIS.2.git \
        "${PROJECT_DIR}/third_party/TRELLIS.2"
else
    echo "  ERROR: No tar at ${REFERENCE_TAR}."
    return 1 2>/dev/null || exit 1
fi

# o-voxel: Trellis2 的体素化工具（将稀疏结构转为网格）
if ! "${VENV}/bin/python" -c "import o_voxel" 2>/dev/null; then
    OVOXEL_DIR="${PROJECT_DIR}/third_party/TRELLIS.2/o-voxel"
    if [ -d "${OVOXEL_DIR}" ]; then
        echo "  Installing o-voxel..."
        uv pip install --python "${VENV}/bin/python" \
            "${OVOXEL_DIR}" --no-build-isolation --no-deps 2>&1 | tail -1
    fi
else
    echo "  o-voxel already installed"
fi

# ============================================================================
# [6/7] 预训练权重 + 数据集（→ /local-ssd/ → symlink）
# ============================================================================
# 为什么放 /local-ssd/ 而不是直接读 S3 FUSE：
#   1. /local-ssd/ 是 NVMe 高速盘，随机 IO 比 FUSE 快 100x
#   2. 模型加载需要大量小文件随机读（safetensors index、config.json 等）
#   3. tar 管道一次性顺序读 S3 再展开到本地盘是最高效的方式
#
# 通过 symlink 让项目代码中的相对路径（pretrained_weights/TRELLIS.2-4B）仍然有效。
echo "=== [6/7] Pretrained weights & dataset ==="
mkdir -p "${WEIGHTS_LOCAL}"

# TRELLIS.2-4B: Shape 和 Tex 两个 Flow Model 的权重（~16 GB）
if [ -d "${WEIGHTS_LOCAL}/TRELLIS.2-4B" ]; then
    echo "  TRELLIS.2-4B: present"
elif s5cmd ls "${TRELLIS2_TAR}" &>/dev/null; then
    echo "  TRELLIS.2-4B: restoring (~30s)..."
    s5cmd cat "${TRELLIS2_TAR}" | tar xf - -C "${WEIGHTS_LOCAL}/"
    echo "  TRELLIS.2-4B: done"
else
    echo "  WARNING: No TRELLIS.2-4B tar. Model loading will fail."
fi

# DINOv3: 图像条件编码器（Trellis2 用 DINOv3 提取图像特征作为条件输入）
if [ -d "${WEIGHTS_LOCAL}/dinov3-vitl16-pretrain-lvd1689m" ]; then
    echo "  DINOv3: present"
elif s5cmd ls "${DINOV3_TAR}" &>/dev/null; then
    echo "  DINOv3: restoring..."
    s5cmd cat "${DINOV3_TAR}" | tar xf - -C "${WEIGHTS_LOCAL}/"
    echo "  DINOv3: done"
fi

# TRELLIS-image-large: 原版 TRELLIS v1 共享组件（~3.1 GB）
# TRELLIS.2-4B 的 pipeline.json 引用了 "microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16"
# —— sparse_structure_decoder 在 v2 中未改动，所以没打包进 TRELLIS.2-4B。
# 不预下载的话，7 个 rank 会同时从 HF 下载，拖慢启动且可能 rate-limit。
if [ -d "${WEIGHTS_LOCAL}/TRELLIS-image-large" ]; then
    echo "  TRELLIS-image-large: present"
elif s5cmd ls "${TRELLIS1_TAR}" &>/dev/null; then
    echo "  TRELLIS-image-large: restoring..."
    s5cmd cat "${TRELLIS1_TAR}" | tar xf - -C "${WEIGHTS_LOCAL}/"
    echo "  TRELLIS-image-large: done"
fi

# Symlink: 让 pipeline.py 的本地路径检查通过
# pipeline.py 拼接 path="pretrained_weights/TRELLIS.2-4B" + v="microsoft/TRELLIS-image-large/ckpts/..."
# 需要 pretrained_weights/TRELLIS.2-4B/microsoft/TRELLIS-image-large/ 指向实际目录
mkdir -p "${WEIGHTS_LOCAL}/TRELLIS.2-4B/microsoft"
ln -sfn "${WEIGHTS_LOCAL}/TRELLIS-image-large" "${WEIGHTS_LOCAL}/TRELLIS.2-4B/microsoft/TRELLIS-image-large"

# 创建 symlink：项目内 pretrained_weights/ → /local-ssd/pretrained_weights/
# 这样训练配置中的 model_name_or_path: "pretrained_weights/TRELLIS.2-4B" 能正确解析
ln -sfn "${WEIGHTS_LOCAL}" "${PROJECT_DIR}/pretrained_weights"

# alphaimages_v2: 训练数据集（2396 张 RGBA 图片 + caption prompt，~478 MB）
# tar 内部结构: alphaimages_v2_formatted/{train.jsonl, test.jsonl, images/}
if [ -d "${DATASET_LOCAL}/images" ]; then
    echo "  Dataset: present"
elif s5cmd ls "${DATASET_TAR}" &>/dev/null; then
    echo "  Dataset: restoring..."
    s5cmd cat "${DATASET_TAR}" | tar xf - -C /local-ssd/
    echo "  Dataset: done"
fi
mkdir -p "${PROJECT_DIR}/dataset"
# dataset/trellis2/ → 直接 symlink 到解压目录（已含 train.jsonl + images/）
ln -sfn "${DATASET_LOCAL}" "${PROJECT_DIR}/dataset/trellis2"

# ============================================================================
# [7/7] 后台 S3 sync（训练产出持久化）
# ============================================================================
# Pod 的 /local-ssd/ 在 Pod 退出后丢失。后台进程每 5 分钟将训练产出
# （checkpoint、日志、可视化）通过 aws s3 sync 同步到 S3。
# - 使用 S3 API（不走 FUSE），避免 rename/覆盖等 FUSE 限制
# - --exclude '*.bin': 跳过大型 optimizer state 文件（只保存模型权重）
# - trap EXIT: Pod 退出前做最后一次同步，确保最新数据不丢
# - --fast 模式跳过（debug 时手动操作更灵活）
echo "=== [7/7] Background S3 sync ==="
if [ "$FAST_MODE" = true ]; then
    echo "  SKIPPED (--fast mode, manual sync if needed)"
else
    SAVES_LOCAL="${PROJECT_DIR}/saves"
    SAVES_S3="${S3_BUCKET}/experiments/flow-factory-trellis2"

    sync_saves() {
        if [ -d "${SAVES_LOCAL}" ]; then
            aws s3 sync "${SAVES_LOCAL}/" "${SAVES_S3}/" \
                --exclude '*.bin' --quiet >> /tmp/ff_s3_sync.log 2>&1 || true
        fi
    }

    (while true; do sleep 300; sync_saves; done) &
    SYNC_PID=$!
    trap "kill ${SYNC_PID} 2>/dev/null || true; sync_saves" EXIT

    echo "  PID: ${SYNC_PID} (every 5 min)"
    echo "  ${SAVES_LOCAL}/ -> ${SAVES_S3}/"
fi

# ============================================================================
# 完成 — 打印环境摘要
# ============================================================================
echo ""
echo "========================================="
echo "  Flow-Factory (trellis2) — Setup Complete"
echo "========================================="
echo "Python:    $(${VENV}/bin/python --version 2>&1)"
echo "Torch:     $(${VENV}/bin/python -c 'import torch; print(torch.__version__, torch.version.cuda)' 2>/dev/null || echo 'FAILED')"
echo "GPUs:      $(${VENV}/bin/python -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo '?')"
echo "Weights:   ${PROJECT_DIR}/pretrained_weights -> ${WEIGHTS_LOCAL}"
echo "Dataset:   ${PROJECT_DIR}/dataset/trellis2 -> ${DATASET_LOCAL}"
echo "TRELLIS.2: ${PROJECT_DIR}/third_party/TRELLIS.2"
echo ""
echo "Quick start:"
echo "  ff-train examples/grpo/lora/trellis2/shape.yaml"
echo ""
echo "========================================="
