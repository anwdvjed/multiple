# 多模态交互模型训练数据集推荐

为您的多分支流多模态模型（视觉、文本、音频）推荐以下数据集，按用途和规模分类。

---

## 一、图像-文本数据集

### 1. MS-COCO Captions (入门推荐)
| 属性 | 详情 |
|------|------|
| **规模** | 33万张图像，150万条描述 |
| **特点** | 高质量人工标注，每张图5条描述 |
| **下载** | https://cocodataset.org |
| **License** | CC BY 4.0 |
| **适用** | 预训练、微调、评估基准 |

### 2. Conceptual Captions (CC3M/CC12M)
| 属性 | 详情 |
|------|------|
| **规模** | CC3M: 330万对 / CC12M: 1200万对 |
| **特点** | 网络爬取，多样性高，自动清洗 |
| **下载** | https://github.com/google-research-datasets/conceptual-captions |
| **License** | 研究用途 |
| **适用** | 大规模预训练 |

### 3. LAION-COCO
| 属性 | 详情 |
|------|------|
| **规模** | 6亿图文对 |
| **特点** | BLIP生成的合成描述，COCO风格 |
| **下载** | https://laion.ai/blog/laion-coco/ |
| **License** | 研究用途 |
| **适用** | 超大规模预训练 |

### 4. Visual Genome
| 属性 | 详情 |
|------|------|
| **规模** | 10.8万图像，540万区域描述，170万VQA |
| **特点** | 密集标注，含关系、属性、区域描述 |
| **下载** | https://huggingface.co/datasets/visual_genome |
| **License** | CC BY 4.0 |
| **适用** | 细粒度视觉理解 |

### 5. LLaVA-CC3M-Pretrain-595K
| 属性 | 详情 |
|------|------|
| **规模** | 59.5万图文对 |
| **特点** | 从CC3M筛选，适合轻量预训练 |
| **下载** | https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K |
| **适用** | 快速预训练实验 |

---

## 二、音频-文本数据集

### 1. AudioCaps (推荐)
| 属性 | 详情 |
|------|------|
| **规模** | 4.6万音频片段 |
| **特点** | 来自AudioSet，人工标注描述 |
| **下载** | https://audiocaps.github.io/ |
| **License** | CC BY 4.0 |
| **适用** | 音频描述生成、音频-文本对齐 |

```python
# PyTorch 加载示例
from aac_datasets import AudioCaps
dataset = AudioCaps(root=".", download=True)
```

### 2. Clotho
| 属性 | 详情 |
|------|------|
| **规模** | 4,981音频，24,905描述 |
| **特点** | 15-30秒音频，每条5个描述，高质量 |
| **下载** | https://zenodo.org/record/3490684 |
| **License** | 研究用途 |
| **适用** | 音频描述、跨模态检索 |

### 3. WavCaps
| 属性 | 详情 |
|------|------|
| **规模** | 40万+音频描述对 |
| **特点** | ChatGPT辅助生成描述，大规模 |
| **下载** | https://github.com/XinhaoMei/WavCaps |
| **适用** | 音频-语言预训练 |

### 4. LibriSpeech (语音识别)
| 属性 | 详情 |
|------|------|
| **规模** | 1000小时英语语音 |
| **特点** | 有声书朗读，干净录音 |
| **下载** | https://www.openslr.org/12 |
| **License** | CC BY 4.0 |
| **适用** | 语音识别、语音-文本对齐 |

---

## 三、视频-音频-文本数据集 (多模态融合)

### 1. VoxCeleb / VoxCeleb2 ⭐推荐
| 属性 | 详情 |
|------|------|
| **规模** | VoxCeleb1: 15万+语段 / VoxCeleb2: 100万+语段 |
| **特点** | 音视频同步，名人访谈，说话人识别 |
| **下载** | https://www.robots.ox.ac.uk/~vgg/data/voxceleb/ |
| **HuggingFace** | https://huggingface.co/datasets/ProgramComputer/voxceleb |
| **适用** | 音视频同步、说话人识别、多模态学习 |

### 2. HowTo100M ⭐大规模推荐
| 属性 | 详情 |
|------|------|
| **规模** | 1.36亿视频片段，120万视频 |
| **特点** | 教学视频，ASR自动转录，23K任务类型 |
| **下载** | https://www.di.ens.fr/willow/research/howto100m/ |
| **适用** | 大规模视频-文本预训练 |

### 3. MSR-VTT
| 属性 | 详情 |
|------|------|
| **规模** | 1万视频片段，20万描述 |
| **特点** | 20个类别，每视频20条人工描述 |
| **下载** | https://huggingface.co/datasets/friedrichor/MSR-VTT |
| **适用** | 视频描述、视频-文本检索基准 |

### 4. ActivityNet Captions
| 属性 | 详情 |
|------|------|
| **规模** | 2万视频，10万句子 |
| **特点** | 长视频，时序密集描述，人类活动 |
| **下载** | http://activity-net.org/download.html |
| **适用** | 时序理解、密集描述 |

### 5. WebVid-2M/10M
| 属性 | 详情 |
|------|------|
| **规模** | 2M/10M视频-文本对 |
| **特点** | 网络爬取，弱监督描述 |
| **下载** | https://github.com/m-bain/webvid |
| **适用** | 视频-文本预训练 |

---

## 四、推荐训练方案

### 方案A：轻量级（适合20GB显存限制）

```
阶段1：单模态预训练
├── 视觉: LLaVA-CC3M-595K (60万对)
├── 文本: 使用预训练词嵌入
└── 音频: Clotho + AudioCaps (~5万对)

阶段2：跨模态对齐
├── 图文对齐: CC3M子集 (100万对)
└── 音文对齐: WavCaps子集 (10万对)

阶段3：多模态融合微调
└── VoxCeleb2子集 (音视频同步)
```

**预计数据量**: ~200万样本  
**训练时间**: 2-3天 (单卡)

### 方案B：中等规模

```
阶段1：大规模预训练
├── 视觉分支: CC12M (1200万对)
├── 音频分支: WavCaps + AudioCaps (50万对)
└── 文本分支: 使用预训练LLM

阶段2：多模态对齐
├── HowTo100M子集 (500万片段)
└── MSR-VTT (20万对)

阶段3：下游任务微调
└── 特定任务数据集
```

**预计数据量**: ~2000万样本  
**训练时间**: 1-2周 (4卡)

---

## 五、数据集下载工具

### img2dataset (图像数据集)
```bash
pip install img2dataset

# 下载CC3M
img2dataset --url_list cc3m.tsv --output_folder cc3m \
    --processes_count 16 --thread_count 64 --image_size 256
```

### aac-datasets (音频数据集)
```bash
pip install aac-datasets

# 下载Clotho和AudioCaps
aac-datasets-download clotho
aac-datasets-download audiocaps
```

### yt-dlp (视频数据集)
```bash
pip install yt-dlp

# 下载YouTube视频
yt-dlp --extract-audio --audio-format mp3 <video_url>
```

---

## 六、数据预处理建议

### 图像预处理
```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])
```

### 音频预处理
```python
import torchaudio

def process_audio(waveform, sample_rate, target_sr=16000):
    # 重采样
    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
        waveform = resampler(waveform)
    
    # 提取Mel频谱
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sr,
        n_fft=1024,
        hop_length=256,
        n_mels=80
    )
    mel_spec = mel_transform(waveform)
    return mel_spec
```

### 文本预处理
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def process_text(text, max_length=256):
    return tokenizer(
        text,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
```

---

## 七、数据集对比总结

| 数据集 | 模态 | 规模 | 质量 | 下载难度 | 推荐指数 |
|--------|------|------|------|----------|----------|
| MS-COCO | 图+文 | 33万 | ⭐⭐⭐⭐⭐ | 简单 | ⭐⭐⭐⭐⭐ |
| CC3M | 图+文 | 330万 | ⭐⭐⭐⭐ | 中等 | ⭐⭐⭐⭐⭐ |
| LAION-COCO | 图+文 | 6亿 | ⭐⭐⭐ | 困难 | ⭐⭐⭐⭐ |
| AudioCaps | 音+文 | 4.6万 | ⭐⭐⭐⭐⭐ | 中等 | ⭐⭐⭐⭐⭐ |
| Clotho | 音+文 | 5千 | ⭐⭐⭐⭐⭐ | 简单 | ⭐⭐⭐⭐ |
| VoxCeleb2 | 视+音 | 100万 | ⭐⭐⭐⭐ | 中等 | ⭐⭐⭐⭐⭐ |
| HowTo100M | 视+音+文 | 1.36亿 | ⭐⭐⭐ | 困难 | ⭐⭐⭐⭐ |
| MSR-VTT | 视+文 | 1万 | ⭐⭐⭐⭐⭐ | 简单 | ⭐⭐⭐⭐ |

---

## 八、开始建议

**如果您刚开始实验**，建议从以下组合开始：

1. **图像-文本**: MS-COCO + LLaVA-CC3M-595K
2. **音频-文本**: Clotho + AudioCaps  
3. **视频/音视频**: VoxCeleb2子集 + MSR-VTT

这个组合总数据量约200万样本，可以在20GB显存限制下完成训练，同时涵盖所有三种模态的预训练和对齐任务。

---

*最后更新: 2026年1月*
