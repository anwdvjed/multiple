#!/bin/bash
# ============================================================
# 多模态模型训练一键运行脚本
# run_all.sh
# 
# 完整流程:
# 1. 安装环境
# 2. 下载数据集
# 3. 训练模型
# 4. 生成报告
# ============================================================

set -e

# 颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $1"; }
warn() { echo -e "${YELLOW}[警告]${NC} $1"; }
error() { echo -e "${RED}[错误]${NC} $1"; exit 1; }

# 默认参数
DATA_ROOT="./data"
OUTPUT_DIR="./outputs"
MODEL_SIZE="base"
BATCH_SIZE=8
EPOCHS=100
SKIP_INSTALL=false
SKIP_DOWNLOAD=false
DATASETS="coco clotho"

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --data-root) DATA_ROOT="$2"; shift 2;;
        --output-dir) OUTPUT_DIR="$2"; shift 2;;
        --model-size) MODEL_SIZE="$2"; shift 2;;
        --batch-size) BATCH_SIZE="$2"; shift 2;;
        --epochs) EPOCHS="$2"; shift 2;;
        --datasets) DATASETS="$2"; shift 2;;
        --skip-install) SKIP_INSTALL=true; shift;;
        --skip-download) SKIP_DOWNLOAD=true; shift;;
        -h|--help)
            echo "使用方法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --data-root DIR      数据目录 (默认: ./data)"
            echo "  --output-dir DIR     输出目录 (默认: ./outputs)"
            echo "  --model-size SIZE    模型规模: tiny/small/base/large (默认: base)"
            echo "  --batch-size N       批次大小 (默认: 8)"
            echo "  --epochs N           训练轮数 (默认: 100)"
            echo "  --datasets LIST      数据集列表 (默认: 'coco clotho')"
            echo "  --skip-install       跳过环境安装"
            echo "  --skip-download      跳过数据下载"
            echo "  -h, --help           显示帮助"
            exit 0;;
        *) error "未知参数: $1";;
    esac
done

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║          多模态交互模型训练系统                            ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "配置:"
echo "  数据目录:   $DATA_ROOT"
echo "  输出目录:   $OUTPUT_DIR"
echo "  模型规模:   $MODEL_SIZE"
echo "  批次大小:   $BATCH_SIZE"
echo "  训练轮数:   $EPOCHS"
echo "  数据集:     $DATASETS"
echo ""

# 记录开始时间
START_TIME=$(date +%s)

# ==================== 步骤1: 安装环境 ====================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "步骤 1/4: 安装环境"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ "$SKIP_INSTALL" = true ]; then
    warn "跳过环境安装"
else
    # 检查是否在虚拟环境中
    if [ -z "$VIRTUAL_ENV" ]; then
        if [ -d "venv" ]; then
            log "激活现有虚拟环境..."
            source venv/bin/activate
        else
            log "创建虚拟环境..."
            python3 -m venv venv
            source venv/bin/activate
        fi
    fi
    
    log "安装依赖..."
    pip install --upgrade pip -q
    pip install -r requirements.txt -q 2>/dev/null || {
        warn "部分依赖安装失败，尝试单独安装核心依赖..."
        pip install torch torchvision torchaudio -q
        pip install transformers numpy pandas tqdm pyyaml tensorboard -q
    }
    
    log "环境安装完成"
fi

# ==================== 步骤2: 下载数据 ====================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "步骤 2/4: 下载数据集"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ "$SKIP_DOWNLOAD" = true ]; then
    warn "跳过数据下载"
else
    # 检查数据是否已存在
    DATA_EXISTS=false
    for ds in $DATASETS; do
        if [ -d "$DATA_ROOT/$ds" ] && [ "$(ls -A $DATA_ROOT/$ds 2>/dev/null)" ]; then
            DATA_EXISTS=true
            log "发现已存在数据: $ds"
        fi
    done
    
    if [ "$DATA_EXISTS" = true ]; then
        read -p "数据已存在，是否重新下载? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log "使用现有数据"
        else
            log "开始下载数据集..."
            python download_datasets.py --datasets $DATASETS --data-root "$DATA_ROOT"
        fi
    else
        log "开始下载数据集..."
        python download_datasets.py --datasets $DATASETS --data-root "$DATA_ROOT"
    fi
fi

# ==================== 步骤3: 训练模型 ====================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "步骤 3/4: 训练模型"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

log "开始训练..."
log "模型: $MODEL_SIZE | 批次: $BATCH_SIZE | 轮数: $EPOCHS"

python run_training.py \
    --data-root "$DATA_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --model-size "$MODEL_SIZE" \
    --batch-size "$BATCH_SIZE" \
    --epochs "$EPOCHS"

TRAIN_STATUS=$?

if [ $TRAIN_STATUS -ne 0 ]; then
    error "训练失败，退出码: $TRAIN_STATUS"
fi

log "训练完成"

# ==================== 步骤4: 生成报告 ====================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "步骤 4/4: 生成最终报告"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 计算总时间
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
HOURS=$((TOTAL_TIME / 3600))
MINUTES=$(((TOTAL_TIME % 3600) / 60))
SECONDS=$((TOTAL_TIME % 60))

# 生成最终摘要
SUMMARY_FILE="$OUTPUT_DIR/training_summary.txt"

cat > "$SUMMARY_FILE" << EOF
╔══════════════════════════════════════════════════════════════╗
║               多模态模型训练完成摘要                            ║
╚══════════════════════════════════════════════════════════════╝

完成时间: $(date)
总耗时: ${HOURS}小时 ${MINUTES}分钟 ${SECONDS}秒

配置信息:
  模型规模:     $MODEL_SIZE
  批次大小:     $BATCH_SIZE
  训练轮数:     $EPOCHS
  数据集:       $DATASETS

输出文件:
  模型权重:     $OUTPUT_DIR/checkpoints/best_model.pt
  最终模型:     $OUTPUT_DIR/checkpoints/final_model.pt
  训练日志:     $OUTPUT_DIR/logs/tensorboard/
  训练报告:     $OUTPUT_DIR/reports/

后续步骤:
  1. 查看训练曲线:
     tensorboard --logdir=$OUTPUT_DIR/logs/tensorboard

  2. 加载模型进行推理:
     python -c "
     import torch
     from model import create_model
     model = create_model('$MODEL_SIZE')
     model.load_state_dict(torch.load('$OUTPUT_DIR/checkpoints/best_model.pt')['model_state_dict'])
     model.eval()
     "

  3. 查看详细报告:
     cat $OUTPUT_DIR/reports/*.md
EOF

cat "$SUMMARY_FILE"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${GREEN}✓ 全部流程完成!${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
