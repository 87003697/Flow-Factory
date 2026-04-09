#!/bin/bash
# Flow-Factory Installation Script
# Usage: bash scripts/setup/install.sh [options]
#
# Options:
#   --name NAME       Conda environment name (default: flow-factory)
#   --python VERSION  Python version (default: 3.10)
#   --deepspeed       Install DeepSpeed
#   --wandb           Install Weights & Biases
#   --swanlab         Install SwanLab
#   --all             Install all optional dependencies
#   --help            Show this help message

set -e

# Default values
ENV_NAME="flow-factory"
PYTHON_VERSION="3.10"
INSTALL_DEEPSPEED=false
INSTALL_WANDB=true
INSTALL_SWANLAB=false
INSTALL_ALL=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --name)
            ENV_NAME="$2"
            shift 2
            ;;
        --python)
            PYTHON_VERSION="$2"
            shift 2
            ;;
        --deepspeed)
            INSTALL_DEEPSPEED=true
            shift
            ;;
        --wandb)
            INSTALL_WANDB=true
            shift
            ;;
        --swanlab)
            INSTALL_SWANLAB=true
            shift
            ;;
        --all)
            INSTALL_ALL=true
            shift
            ;;
        --help)
            head -n 13 "$0" | tail -n 12
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "  Flow-Factory Installation Script"
echo "=========================================="
echo ""
echo "Environment name: $ENV_NAME"
echo "Python version: $PYTHON_VERSION"
echo ""

# Get project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH"
    exit 1
fi

# Create conda environment
echo "[1/4] Creating conda environment: $ENV_NAME (Python $PYTHON_VERSION)..."
if conda env list | grep -q "^$ENV_NAME "; then
    echo "  Environment '$ENV_NAME' already exists. Skipping creation."
else
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
fi

# Initialize conda for this shell
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

echo "[2/4] Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo "[3/4] Installing Flow-Factory..."
cd "$PROJECT_ROOT"
pip install -e .

echo "[4/4] Installing optional dependencies..."
if $INSTALL_ALL; then
    pip install -e ".[all]"
    pip install wandb swanlab
    echo "  Installed: all optional dependencies"
else
    if $INSTALL_DEEPSPEED; then
        pip install -e ".[deepspeed]"
        echo "  Installed: deepspeed"
    fi
    if $INSTALL_WANDB; then
        pip install -e ".[wandb]"
        echo "  Installed: wandb"
    fi
    if $INSTALL_SWANLAB; then
        pip install -e ".[swanlab]"
        echo "  Installed: swanlab"
    fi
fi

echo ""
echo "=========================================="
echo "  Installation Complete!"
echo "=========================================="
echo ""
echo "To activate the environment, run:"
echo "  conda activate $ENV_NAME"
echo ""
echo "To start training, run:"
echo "  ff-train examples/grpo/lora/flux1.yaml"
echo ""
