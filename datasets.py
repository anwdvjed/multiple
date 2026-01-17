"""
多模态数据集加载器
datasets.py

支持的数据集:
- MS-COCO Captions (图像-文本)
- Clotho (音频-文本)
- AudioCaps (音频-文本)
- VoxCeleb (音频-视频)
- 混合多模态数据集
"""

import os
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
from PIL import Image

try:
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False

try:
    from torchvision import transforms
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


@dataclass
class DataConfig:
    """数据配置"""
    # 图像
    img_size: int = 224
    img_mean: Tuple[float, ...] = (0.485, 0.456, 0.406)
    img_std: Tuple[float, ...] = (0.229, 0.224, 0.225)
    
    # 音频
    sample_rate: int = 16000
    n_mels: int = 80
    n_fft: int = 1024
    hop_length: int = 256
    max_audio_len: int = 10  # 秒
    
    # 文本
    max_text_len: int = 256
    tokenizer_name: str = "bert-base-uncased"


class TextProcessor:
    """文本处理器"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.tokenizer = None
        
        if TRANSFORMERS_AVAILABLE:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
            except Exception as e:
                print(f"警告: 无法加载tokenizer: {e}")
    
    def __call__(self, text: str) -> Dict[str, torch.Tensor]:
        if self.tokenizer is None:
            # 简单的字符级tokenization作为备选
            tokens = [ord(c) % 32000 for c in text[:self.config.max_text_len]]
            tokens = tokens + [0] * (self.config.max_text_len - len(tokens))
            return {
                'input_ids': torch.tensor(tokens, dtype=torch.long),
                'attention_mask': torch.tensor([1] * len(text) + [0] * (self.config.max_text_len - len(text)), dtype=torch.long)
            }
        
        encoded = self.tokenizer(
            text,
            max_length=self.config.max_text_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0)
        }


class ImageProcessor:
    """图像处理器"""
    
    def __init__(self, config: DataConfig, is_train: bool = True):
        self.config = config
        
        if is_train:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(config.img_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.1, 0.1, 0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=config.img_mean, std=config.img_std)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(int(config.img_size * 1.14)),
                transforms.CenterCrop(config.img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=config.img_mean, std=config.img_std)
            ])
    
    def __call__(self, image: Union[str, Path, Image.Image]) -> torch.Tensor:
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        return self.transform(image)


class AudioProcessor:
    """音频处理器"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.max_samples = config.sample_rate * config.max_audio_len
        
        if TORCHAUDIO_AVAILABLE:
            self.mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=config.sample_rate,
                n_fft=config.n_fft,
                hop_length=config.hop_length,
                n_mels=config.n_mels
            )
    
    def __call__(self, audio_path: Union[str, Path]) -> torch.Tensor:
        if not TORCHAUDIO_AVAILABLE:
            # 返回随机数据作为占位
            max_frames = self.max_samples // self.config.hop_length
            return torch.randn(max_frames, self.config.n_mels)
        
        # 加载音频
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # 转单声道
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # 重采样
        if sample_rate != self.config.sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.config.sample_rate)
            waveform = resampler(waveform)
        
        # 截断或填充
        if waveform.shape[1] > self.max_samples:
            waveform = waveform[:, :self.max_samples]
        elif waveform.shape[1] < self.max_samples:
            padding = self.max_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        
        # 提取Mel频谱
        mel_spec = self.mel_transform(waveform)
        
        # 转换为 [T, F] 格式
        mel_spec = mel_spec.squeeze(0).transpose(0, 1)
        
        # 对数变换
        mel_spec = torch.log(mel_spec + 1e-9)
        
        return mel_spec


# ==================== 具体数据集实现 ====================

class COCOCaptionDataset(Dataset):
    """MS-COCO Captions 数据集"""
    
    def __init__(
        self,
        root: str,
        split: str = 'train',
        config: DataConfig = None
    ):
        self.root = Path(root)
        self.split = split
        self.config = config or DataConfig()
        
        # 处理器
        self.image_processor = ImageProcessor(self.config, is_train=(split == 'train'))
        self.text_processor = TextProcessor(self.config)
        
        # 加载标注
        self.samples = self._load_annotations()
    
    def _load_annotations(self) -> List[Dict]:
        """加载COCO标注"""
        split_name = 'train2017' if self.split == 'train' else 'val2017'
        ann_file = self.root / 'annotations' / f'captions_{split_name}.json'
        
        if not ann_file.exists():
            print(f"警告: 标注文件不存在 {ann_file}")
            return []
        
        with open(ann_file, 'r') as f:
            data = json.load(f)
        
        # 构建图像ID到文件名的映射
        id_to_filename = {img['id']: img['file_name'] for img in data['images']}
        
        # 构建样本列表
        samples = []
        for ann in data['annotations']:
            img_id = ann['image_id']
            if img_id in id_to_filename:
                samples.append({
                    'image_path': self.root / split_name / id_to_filename[img_id],
                    'caption': ann['caption'],
                    'image_id': img_id
                })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # 加载图像
        image = self.image_processor(sample['image_path'])
        
        # 处理文本
        text_data = self.text_processor(sample['caption'])
        
        return {
            'images': image,
            'input_ids': text_data['input_ids'],
            'attention_mask': text_data['attention_mask'],
            'modalities': ['vision', 'text']
        }


class ClothoDataset(Dataset):
    """Clotho 音频描述数据集"""
    
    def __init__(
        self,
        root: str,
        split: str = 'development',
        config: DataConfig = None
    ):
        self.root = Path(root)
        self.split = split
        self.config = config or DataConfig()
        
        # 处理器
        self.audio_processor = AudioProcessor(self.config)
        self.text_processor = TextProcessor(self.config)
        
        # 加载标注
        self.samples = self._load_annotations()
    
    def _load_annotations(self) -> List[Dict]:
        """加载Clotho标注"""
        # 尝试多种可能的文件路径
        possible_paths = [
            self.root / f'clotho_captions_{self.split}.csv',
            self.root / 'data' / f'clotho_captions_{self.split}.csv',
            self.root / f'{self.split}.csv'
        ]
        
        csv_path = None
        for path in possible_paths:
            if path.exists():
                csv_path = path
                break
        
        if csv_path is None:
            print(f"警告: 找不到Clotho标注文件")
            return []
        
        import csv
        samples = []
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                audio_name = row.get('file_name', row.get('audio', ''))
                
                # 查找音频文件
                audio_path = None
                for subdir in ['', self.split, f'clotho_audio_{self.split}']:
                    candidate = self.root / subdir / audio_name
                    if candidate.exists():
                        audio_path = candidate
                        break
                
                if audio_path is None:
                    continue
                
                # 获取所有描述
                captions = []
                for i in range(1, 6):
                    cap_key = f'caption_{i}'
                    if cap_key in row and row[cap_key]:
                        captions.append(row[cap_key])
                
                if captions:
                    samples.append({
                        'audio_path': audio_path,
                        'captions': captions
                    })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # 加载音频
        audio = self.audio_processor(sample['audio_path'])
        
        # 随机选择一个描述
        caption = random.choice(sample['captions'])
        text_data = self.text_processor(caption)
        
        return {
            'audio': audio,
            'input_ids': text_data['input_ids'],
            'attention_mask': text_data['attention_mask'],
            'modalities': ['audio', 'text']
        }


class AudioCapsDataset(Dataset):
    """AudioCaps 数据集"""
    
    def __init__(
        self,
        root: str,
        split: str = 'train',
        config: DataConfig = None
    ):
        self.root = Path(root)
        self.split = split
        self.config = config or DataConfig()
        
        self.audio_processor = AudioProcessor(self.config)
        self.text_processor = TextProcessor(self.config)
        
        self.samples = self._load_annotations()
    
    def _load_annotations(self) -> List[Dict]:
        """加载AudioCaps标注"""
        csv_path = self.root / f'{self.split}.csv'
        
        if not csv_path.exists():
            print(f"警告: 找不到AudioCaps标注文件 {csv_path}")
            return []
        
        import csv
        samples = []
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                audiocap_id = row.get('audiocap_id', '')
                youtube_id = row.get('youtube_id', '')
                caption = row.get('caption', '')
                
                # 查找音频文件
                audio_path = self.root / 'audio' / f'{youtube_id}.wav'
                if not audio_path.exists():
                    audio_path = self.root / 'audio' / f'{audiocap_id}.wav'
                
                if audio_path.exists() and caption:
                    samples.append({
                        'audio_path': audio_path,
                        'caption': caption
                    })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        audio = self.audio_processor(sample['audio_path'])
        text_data = self.text_processor(sample['caption'])
        
        return {
            'audio': audio,
            'input_ids': text_data['input_ids'],
            'attention_mask': text_data['attention_mask'],
            'modalities': ['audio', 'text']
        }


class VoxCelebDataset(Dataset):
    """VoxCeleb 音视频数据集"""
    
    def __init__(
        self,
        root: str,
        split: str = 'dev',
        config: DataConfig = None
    ):
        self.root = Path(root)
        self.split = split
        self.config = config or DataConfig()
        
        self.image_processor = ImageProcessor(self.config, is_train=(split == 'dev'))
        self.audio_processor = AudioProcessor(self.config)
        
        self.samples = self._load_samples()
    
    def _load_samples(self) -> List[Dict]:
        """扫描VoxCeleb目录"""
        samples = []
        
        split_dir = self.root / self.split / 'mp4'
        if not split_dir.exists():
            split_dir = self.root / self.split
        
        if not split_dir.exists():
            print(f"警告: VoxCeleb目录不存在 {split_dir}")
            return []
        
        # 遍历说话人目录
        for speaker_dir in split_dir.iterdir():
            if not speaker_dir.is_dir():
                continue
            
            # 遍历视频目录
            for video_dir in speaker_dir.iterdir():
                if not video_dir.is_dir():
                    continue
                
                # 遍历片段
                for video_file in video_dir.glob('*.mp4'):
                    samples.append({
                        'video_path': video_file,
                        'speaker_id': speaker_dir.name
                    })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # 提取帧和音频 (简化版本)
        # 实际应用中需要使用opencv或decord提取视频帧
        
        # 这里返回占位数据
        image = torch.randn(3, self.config.img_size, self.config.img_size)
        audio = torch.randn(self.config.max_audio_len * self.config.sample_rate // self.config.hop_length, self.config.n_mels)
        
        return {
            'images': image,
            'audio': audio,
            'modalities': ['vision', 'audio']
        }


# ==================== 混合数据集 ====================

class MultiModalDataset(Dataset):
    """
    混合多模态数据集
    支持从多个数据集中采样
    """
    
    def __init__(
        self,
        datasets: Dict[str, Dataset],
        weights: Optional[Dict[str, float]] = None,
        total_samples: Optional[int] = None
    ):
        self.datasets = datasets
        self.dataset_names = list(datasets.keys())
        self.dataset_sizes = {name: len(ds) for name, ds in datasets.items()}
        
        # 计算采样权重
        if weights is None:
            weights = {name: 1.0 for name in self.dataset_names}
        self.weights = weights
        
        # 归一化权重
        total_weight = sum(weights.values())
        self.normalized_weights = {name: w / total_weight for name, w in weights.items()}
        
        # 总样本数
        self.total_samples = total_samples or sum(self.dataset_sizes.values())
        
        # 预计算每个数据集的采样数量
        self._build_index()
    
    def _build_index(self):
        """构建采样索引"""
        self.index = []
        
        for name, ds in self.datasets.items():
            weight = self.normalized_weights[name]
            n_samples = int(self.total_samples * weight)
            
            indices = list(range(len(ds)))
            if n_samples > len(ds):
                # 重复采样
                indices = indices * (n_samples // len(ds) + 1)
            
            random.shuffle(indices)
            indices = indices[:n_samples]
            
            for idx in indices:
                self.index.append((name, idx))
        
        random.shuffle(self.index)
    
    def __len__(self) -> int:
        return len(self.index)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        dataset_name, sample_idx = self.index[idx]
        sample = self.datasets[dataset_name][sample_idx]
        sample['dataset'] = dataset_name
        return sample


# ==================== 数据加载工具 ====================

def collate_multimodal(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    多模态数据批次整理函数
    处理不同模态的缺失情况
    """
    result = {}
    
    # 收集每个字段
    keys = set()
    for sample in batch:
        keys.update(sample.keys())
    
    for key in keys:
        if key == 'modalities' or key == 'dataset':
            result[key] = [s.get(key) for s in batch]
            continue
        
        values = []
        mask = []
        
        for sample in batch:
            if key in sample and sample[key] is not None:
                values.append(sample[key])
                mask.append(1)
            else:
                mask.append(0)
        
        if values:
            if isinstance(values[0], torch.Tensor):
                # 填充缺失值
                if len(values) < len(batch):
                    dummy = torch.zeros_like(values[0])
                    full_values = []
                    value_idx = 0
                    for m in mask:
                        if m:
                            full_values.append(values[value_idx])
                            value_idx += 1
                        else:
                            full_values.append(dummy)
                    values = full_values
                
                result[key] = torch.stack(values)
            else:
                result[key] = values
            
            result[f'{key}_mask'] = torch.tensor(mask, dtype=torch.bool)
    
    return result


def create_dataloaders(
    data_root: str,
    config: DataConfig,
    batch_size: int = 8,
    num_workers: int = 4,
    datasets_to_use: List[str] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    创建训练和验证数据加载器
    """
    data_root = Path(data_root)
    
    if datasets_to_use is None:
        datasets_to_use = ['coco', 'clotho', 'audiocaps']
    
    train_datasets = {}
    val_datasets = {}
    
    # 加载各数据集
    if 'coco' in datasets_to_use and (data_root / 'coco').exists():
        try:
            train_datasets['coco'] = COCOCaptionDataset(data_root / 'coco', 'train', config)
            val_datasets['coco'] = COCOCaptionDataset(data_root / 'coco', 'val', config)
            print(f"✓ COCO: {len(train_datasets['coco'])} train, {len(val_datasets['coco'])} val")
        except Exception as e:
            print(f"✗ COCO 加载失败: {e}")
    
    if 'clotho' in datasets_to_use and (data_root / 'clotho').exists():
        try:
            train_datasets['clotho'] = ClothoDataset(data_root / 'clotho', 'development', config)
            val_datasets['clotho'] = ClothoDataset(data_root / 'clotho', 'validation', config)
            print(f"✓ Clotho: {len(train_datasets['clotho'])} train, {len(val_datasets['clotho'])} val")
        except Exception as e:
            print(f"✗ Clotho 加载失败: {e}")
    
    if 'audiocaps' in datasets_to_use and (data_root / 'audiocaps').exists():
        try:
            train_datasets['audiocaps'] = AudioCapsDataset(data_root / 'audiocaps', 'train', config)
            val_datasets['audiocaps'] = AudioCapsDataset(data_root / 'audiocaps', 'val', config)
            print(f"✓ AudioCaps: {len(train_datasets['audiocaps'])} train, {len(val_datasets['audiocaps'])} val")
        except Exception as e:
            print(f"✗ AudioCaps 加载失败: {e}")
    
    if 'voxceleb' in datasets_to_use and (data_root / 'voxceleb').exists():
        try:
            train_datasets['voxceleb'] = VoxCelebDataset(data_root / 'voxceleb', 'dev', config)
            val_datasets['voxceleb'] = VoxCelebDataset(data_root / 'voxceleb', 'test', config)
            print(f"✓ VoxCeleb: {len(train_datasets['voxceleb'])} train, {len(val_datasets['voxceleb'])} val")
        except Exception as e:
            print(f"✗ VoxCeleb 加载失败: {e}")
    
    if not train_datasets:
        raise ValueError("没有可用的数据集!")
    
    # 创建混合数据集
    train_dataset = MultiModalDataset(train_datasets)
    val_dataset = MultiModalDataset(val_datasets)
    
    print(f"\n总计: {len(train_dataset)} train, {len(val_dataset)} val")
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_multimodal,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_multimodal,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == '__main__':
    # 测试数据集加载
    config = DataConfig()
    
    print("测试数据加载器...")
    try:
        train_loader, val_loader = create_dataloaders(
            './data',
            config,
            batch_size=4,
            num_workers=0
        )
        
        print(f"\n训练批次数: {len(train_loader)}")
        print(f"验证批次数: {len(val_loader)}")
        
        # 测试一个批次
        batch = next(iter(train_loader))
        print(f"\n批次内容:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
            else:
                print(f"  {key}: {type(value)}")
                
    except Exception as e:
        print(f"测试失败: {e}")
