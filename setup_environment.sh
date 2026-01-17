#!/bin/bash
# ============================================================
# 多模态模型训练环境安装脚本
# setup_environment.sh
# ============================================================

set -e  # 遇错即停

echo "========================================================"
echo "       多模态模型训练环境安装"
echo "========================================================"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ==================== 1. 检查系统环境 ====================
log_info "步骤 1/6: 检查系统环境..."

# 检查Python版本
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
log_info "Python版本: $PYTHON_VERSION"

# 检查CUDA
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
    log_info "GPU: $GPU_NAME"
    log_info "显存: $GPU_MEMORY"
    log_info "驱动版本: $CUDA_VERSION"
else
    log_warn "未检测到NVIDIA GPU，将使用CPU训练"
fi

# 检查磁盘空间
DISK_SPACE=$(df -h . | awk 'NR==2 {print $4}')
log_info "可用磁盘空间: $DISK_SPACE"

# ==================== 2. 创建虚拟环境 ====================
log_info "步骤 2/6: 创建Python虚拟环境..."

VENV_DIR="./venv"

if [ -d "$VENV_DIR" ]; then
    log_warn "虚拟环境已存在，跳过创建"
else
    python3 -m venv $VENV_DIR
    log_info "虚拟环境创建成功: $VENV_DIR"
fi

# 激活虚拟环境
source $VENV_DIR/bin/activate
log_info "虚拟环境已激活"

# 升级pip
pip install --upgrade pip setuptools wheel

# ==================== 3. 安装PyTorch ====================
log_info "步骤 3/6: 安装PyTorch..."

# 检测CUDA版本并安装对应PyTorch
if command -v nvidia-smi &> /dev/null; then
    # 获取CUDA版本
    CUDA_VER=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
    log_info "检测到CUDA版本: $CUDA_VER"
    
    # 根据CUDA版本选择PyTorch
    if [[ "$CUDA_VER" == "12."* ]]; then
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    elif [[ "$CUDA_VER" == "11."* ]]; then
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    else
        pip install torch torchvision torchaudio
    fi
else
    # CPU版本
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# 验证PyTorch安装
python3 -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}')"

# ==================== 4. 安装其他依赖 ====================
log_info "步骤 4/6: 安装项目依赖..."

pip install -r requirements.txt

# 安装额外的系统依赖 (如果需要)
if command -v apt-get &> /dev/null; then
    log_info "安装系统依赖..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq ffmpeg libsndfile1 > /dev/null 2>&1 || log_warn "部分系统依赖安装失败，可能需要手动安装"
fi

# ==================== 5. 创建项目目录结构 ====================
log_info "步骤 5/6: 创建项目目录结构..."

# 创建必要的目录
mkdir -p data/{coco,cc3m,clotho,audiocaps,voxceleb}
mkdir -p outputs
mkdir -p checkpoints
mkdir -p logs/{tensorboard,training}
mkdir -p reports

log_info "目录结构创建完成"

# 显示目录结构
echo ""
echo "项目目录结构:"
echo "├── data/"
echo "│   ├── coco/"
echo "│   ├── cc3m/"
echo "│   ├── clotho/"
echo "│   ├── audiocaps/"
echo "│   └── voxceleb/"
echo "├── outputs/"
echo "├── checkpoints/"
echo "├── logs/"
echo "│   ├── tensorboard/"
echo "│   └── training/"
echo "└── reports/"
echo ""

# ==================== 6. 验证安装 ====================
log_info "步骤 6/6: 验证安装..."

python3 << 'EOF'
import sys

def check_import(module_name, package_name=None):
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False

modules = [
    ("torch", "PyTorch"),
    ("torchvision", "TorchVision"),
    ("torchaudio", "TorchAudio"),
    ("transformers", "Transformers"),
    ("PIL", "Pillow"),
    ("cv2", "OpenCV"),
    ("librosa", "Librosa"),
    ("numpy", "NumPy"),
    ("pandas", "Pandas"),
    ("tqdm", "tqdm"),
    ("yaml", "PyYAML"),
]

print("\n依赖检查结果:")
print("-" * 40)

all_ok = True
for module, name in modules:
    status = "✓" if check_import(module) else "✗"
    if status == "✗":
        all_ok = False
    print(f"  {status} {name}")

print("-" * 40)

if all_ok:
    print("所有依赖安装成功!")
else:
    print("部分依赖安装失败，请检查错误信息")
    sys.exit(1)

# 检查GPU
import torch
if torch.cuda.is_available():
    print(f"\nGPU信息:")
    print(f"  设备数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name}")
        print(f"    显存: {props.total_memory / 1024**3:.1f} GB")
else:
    print("\n警告: CUDA不可用，将使用CPU训练")
EOF

echo ""
log_info "========================================================"
log_info "环境安装完成!"
log_info "========================================================"
echo ""
echo "后续步骤:"
echo "  1. 激活虚拟环境: source venv/bin/activate"
echo "  2. 下载数据集: python download_datasets.py"
echo "  3. 开始训练: python run_training.py"
echo ""
