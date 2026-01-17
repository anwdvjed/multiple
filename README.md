# 多模态交互模型训练项目

完整的多模态模型训练流程，包括环境安装、数据下载、模型训练和报告生成。

## 快速开始

### 一键运行（推荐）

```bash
# 给脚本添加执行权限
chmod +x run_all.sh

# 运行完整流程
./run_all.sh

# 或指定参数
./run_all.sh --model-size base --batch-size 8 --epochs 50
```

### 分步执行

```bash
# 1. 安装环境
bash setup_environment.sh

# 2. 激活虚拟环境
source venv/bin/activate

# 3. 下载数据集
python download_datasets.py --datasets coco clotho audiocaps

# 4. 开始训练
python run_training.py --model-size base --batch-size 8 --epochs 100

# 5. 查看训练曲线
tensorboard --logdir=./outputs/logs/tensorboard
```

## 项目结构

```
multimodal_project/
├── setup_environment.sh    # 环境安装脚本
├── download_datasets.py    # 数据集下载脚本
├── run_training.py         # 训练主脚本
├── run_all.sh             # 一键运行脚本
├── config.yaml            # 配置文件
├── requirements.txt       # Python依赖
├── model.py               # 模型定义
├── datasets.py            # 数据集加载器
├── README.md              # 本文档
│
├── data/                  # 数据目录（自动创建）
│   ├── coco/
│   ├── clotho/
│   ├── audiocaps/
│   └── voxceleb/
│
└── outputs/               # 输出目录（自动创建）
    ├── checkpoints/       # 模型权重
    ├── logs/              # 训练日志
    └── reports/           # 训练报告
```

## 硬件要求

| 模型规模 | 参数量 | 最小显存 | 推荐显存 |
|---------|--------|---------|---------|
| tiny    | ~15M   | 4GB     | 8GB     |
| small   | ~35M   | 8GB     | 12GB    |
| base    | ~80M   | 12GB    | 16GB    |
| large   | ~200M  | 20GB    | 24GB    |

## 配置说明

### 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data-root` | `./data` | 数据存储目录 |
| `--output-dir` | `./outputs` | 输出目录 |
| `--model-size` | `base` | 模型规模: tiny/small/base/large |
| `--batch-size` | `8` | 批次大小 |
| `--epochs` | `100` | 训练轮数 |
| `--lr` | `1e-4` | 学习率 |
| `--resume` | `None` | 恢复训练的检查点路径 |

### 配置文件 (config.yaml)

```yaml
model:
  size: base
  embed_dim: 512
  use_checkpoint: true

training:
  epochs: 100
  batch_size: 8
  learning_rate: 1.0e-4
  use_amp: true
  gradient_accumulation_steps: 4

paths:
  data_root: ./data
  output_dir: ./outputs
```

## 数据集

### 支持的数据集

| 数据集 | 模态 | 规模 | 说明 |
|--------|------|------|------|
| MS-COCO | 图+文 | 33万 | 图像描述基准 |
| Clotho | 音+文 | 5千 | 音频描述 |
| AudioCaps | 音+文 | 4.6万 | 音频描述 |
| VoxCeleb | 视+音 | 100万 | 说话人识别 |

### 下载特定数据集

```bash
# 下载单个数据集
python download_datasets.py --datasets coco

# 下载多个数据集
python download_datasets.py --datasets coco clotho audiocaps

# 下载全部
python download_datasets.py --datasets all
```

## 训练流程

### 1. 单模态预训练
- 视觉分支：在COCO图像上预训练
- 文本分支：使用预训练词嵌入
- 音频分支：在Clotho/AudioCaps上预训练

### 2. 跨模态对齐
- 图文对比学习
- 音文对比学习

### 3. 全模型微调
- 解冻所有参数
- 端到端训练

## 显存优化策略

本项目采用多种策略确保在20GB显存内训练：

1. **梯度检查点** - 以计算换显存
2. **混合精度训练 (AMP)** - FP16存储
3. **梯度累积** - 模拟大批次
4. **分块注意力** - 减少峰值显存

## 输出文件

训练完成后，在 `outputs/` 目录下会生成：

```
outputs/
├── checkpoints/
│   ├── best_model.pt       # 最佳验证损失模型
│   ├── final_model.pt      # 最终模型
│   └── checkpoint_epoch_*.pt
├── logs/
│   └── tensorboard/        # TensorBoard日志
└── reports/
    ├── report_*.json       # JSON格式报告
    └── report_*.md         # Markdown格式报告
```

## 加载训练好的模型

```python
import torch
from model import create_model

# 创建模型
model = create_model('base')

# 加载权重
checkpoint = torch.load('outputs/checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 推理
with torch.no_grad():
    outputs = model(
        images=image_tensor,      # [B, 3, 224, 224]
        input_ids=text_ids,       # [B, L]
        attention_mask=text_mask, # [B, L]
        audio=audio_mel           # [B, T, 80]
    )
    
    # 获取融合特征
    features = outputs['fused_features']  # [B, 512]
    
    # 获取分类logits
    logits = outputs['logits']  # [B, num_classes]
```

## 常见问题

### Q: 显存不足怎么办？

A: 尝试以下方法：
1. 减小 `--batch-size`
2. 使用 `--model-size tiny` 或 `small`
3. 增加 `gradient_accumulation_steps` (在config.yaml中)

### Q: 数据集下载失败？

A: 
1. 检查网络连接
2. 某些数据集需要手动下载（如VoxCeleb）
3. 使用 `--skip-existing` 跳过已下载的数据集

### Q: 如何恢复中断的训练？

A: 使用 `--resume` 参数：
```bash
python run_training.py --resume outputs/checkpoints/checkpoint_epoch_50.pt
```

## 许可证

MIT License
