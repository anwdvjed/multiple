#!/usr/bin/env python3
"""
多模态模型训练主脚本
run_training.py
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

from model import create_model, MultiModalInteractionModel
from datasets import DataConfig, create_dataloaders


@dataclass
class TrainingState:
    """训练状态"""
    epoch: int = 0
    global_step: int = 0
    best_val_loss: float = float('inf')
    best_val_acc: float = 0.0
    train_losses: List[float] = None
    val_losses: List[float] = None
    val_accs: List[float] = None
    
    def __post_init__(self):
        self.train_losses = self.train_losses or []
        self.val_losses = self.val_losses or []
        self.val_accs = self.val_accs or []


class Trainer:
    """训练器"""
    
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        train_cfg = config.get('training', {})
        self.epochs = train_cfg.get('epochs', 100)
        self.batch_size = train_cfg.get('batch_size', 8)
        self.lr = train_cfg.get('learning_rate', 1e-4)
        self.min_lr = train_cfg.get('min_lr', 1e-6)
        self.weight_decay = train_cfg.get('weight_decay', 0.05)
        self.warmup_epochs = train_cfg.get('warmup_epochs', 5)
        self.use_amp = train_cfg.get('use_amp', True)
        self.max_grad_norm = train_cfg.get('max_grad_norm', 1.0)
        self.gradient_accumulation_steps = train_cfg.get('gradient_accumulation_steps', 4)
        self.cls_loss_weight = train_cfg.get('cls_loss_weight', 1.0)
        self.contrastive_loss_weight = train_cfg.get('contrastive_loss_weight', 0.5)
        self.temperature = train_cfg.get('temperature', 0.07)
        self.save_every = train_cfg.get('save_every', 5)
        self.log_every = train_cfg.get('log_every', 100)
        
        paths_cfg = config.get('paths', {})
        self.checkpoint_dir = Path(paths_cfg.get('checkpoint_dir', './checkpoints'))
        self.log_dir = Path(paths_cfg.get('log_dir', './logs'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.scaler = GradScaler(enabled=self.use_amp)
        self.state = TrainingState()
        
        self.writer = SummaryWriter(self.log_dir / 'tensorboard') if TENSORBOARD_AVAILABLE else None
    
    def _create_optimizer(self):
        decay, no_decay = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            (no_decay if 'bias' in name or 'norm' in name else decay).append(param)
        return AdamW([{'params': decay, 'weight_decay': self.weight_decay}, {'params': no_decay, 'weight_decay': 0.0}], lr=self.lr)
    
    def _create_scheduler(self):
        warmup_steps = len(self.train_loader) * self.warmup_epochs
        total_steps = len(self.train_loader) * self.epochs
        warmup = LinearLR(self.optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
        cosine = CosineAnnealingLR(self.optimizer, T_max=total_steps - warmup_steps, eta_min=self.min_lr)
        return SequentialLR(self.optimizer, [warmup, cosine], milestones=[warmup_steps])
    
    def compute_loss(self, outputs, batch):
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=self.device)
        
        if 'labels' in batch:
            cls_loss = F.cross_entropy(outputs['logits'], batch['labels'])
            loss_dict['cls_loss'] = cls_loss.item()
            total_loss = total_loss + self.cls_loss_weight * cls_loss
        
        if outputs.get('vision_features') is not None and outputs.get('text_features') is not None:
            v_emb = F.normalize(self.model.contrastive_proj(outputs['vision_features']), dim=-1)
            t_emb = F.normalize(self.model.contrastive_proj(outputs['text_features']), dim=-1)
            c_loss = self.model.get_contrastive_loss(v_emb, t_emb, self.temperature)
            loss_dict['contrastive_loss'] = c_loss.item()
            total_loss = total_loss + self.contrastive_loss_weight * c_loss
        
        if total_loss.item() == 0:
            total_loss = outputs['fused_features'].norm(dim=-1).mean() * 0.01
        
        loss_dict['total_loss'] = total_loss.item()
        return total_loss, loss_dict
    
    def train_epoch(self):
        self.model.train()
        total_loss, num_batches = 0.0, 0
        self.optimizer.zero_grad()
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.state.epoch}')
        for batch_idx, batch in enumerate(pbar):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            with autocast(enabled=self.use_amp):
                outputs = self.model(images=batch.get('images'), input_ids=batch.get('input_ids'),
                                     attention_mask=batch.get('attention_mask'), audio=batch.get('audio'), return_features=True)
                loss, loss_dict = self.compute_loss(outputs, batch)
                loss = loss / self.gradient_accumulation_steps
            
            self.scaler.scale(loss).backward()
            
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()
                self.state.global_step += 1
            
            total_loss += loss_dict['total_loss']
            num_batches += 1
            pbar.set_postfix({'loss': f"{loss_dict['total_loss']:.4f}", 'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"})
            
            if self.writer and batch_idx % self.log_every == 0:
                for k, v in loss_dict.items():
                    self.writer.add_scalar(f'train/{k}', v, self.state.global_step)
        
        return {'avg_loss': total_loss / num_batches}
    
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss, total_correct, total_samples = 0.0, 0, 0
        
        for batch in tqdm(self.val_loader, desc='Validation'):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            with autocast(enabled=self.use_amp):
                outputs = self.model(images=batch.get('images'), input_ids=batch.get('input_ids'),
                                     attention_mask=batch.get('attention_mask'), audio=batch.get('audio'), return_features=True)
                loss, _ = self.compute_loss(outputs, batch)
            
            total_loss += loss.item()
            if 'labels' in batch:
                total_correct += (outputs['logits'].argmax(dim=-1) == batch['labels']).sum().item()
                total_samples += batch['labels'].size(0)
        
        metrics = {'val_loss': total_loss / len(self.val_loader)}
        if total_samples > 0:
            metrics['val_acc'] = total_correct / total_samples
        return metrics
    
    def save_checkpoint(self, filename):
        torch.save({
            'epoch': self.state.epoch, 'global_step': self.state.global_step,
            'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(), 'scaler_state_dict': self.scaler.state_dict(),
            'state': asdict(self.state), 'config': self.config
        }, self.checkpoint_dir / filename)
        print(f"检查点已保存: {self.checkpoint_dir / filename}")
    
    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        self.scaler.load_state_dict(ckpt['scaler_state_dict'])
        self.state.epoch = ckpt.get('state', {}).get('epoch', 0)
        self.state.global_step = ckpt.get('state', {}).get('global_step', 0)
        print(f"从 {path} 加载检查点, epoch: {self.state.epoch}")
    
    def train(self):
        print("=" * 60)
        print(f"开始训练 | 设备: {self.device} | 批次: {self.batch_size} | 梯度累积: {self.gradient_accumulation_steps}")
        print("=" * 60)
        
        start_time = time.time()
        
        for epoch in range(self.state.epoch, self.epochs):
            self.state.epoch = epoch
            
            train_metrics = self.train_epoch()
            self.state.train_losses.append(train_metrics['avg_loss'])
            print(f"\nEpoch {epoch} - Train Loss: {train_metrics['avg_loss']:.4f}")
            
            val_metrics = self.validate()
            self.state.val_losses.append(val_metrics['val_loss'])
            print(f"Val Loss: {val_metrics['val_loss']:.4f}")
            
            if self.writer:
                self.writer.add_scalar('epoch/train_loss', train_metrics['avg_loss'], epoch)
                self.writer.add_scalar('epoch/val_loss', val_metrics['val_loss'], epoch)
            
            if val_metrics['val_loss'] < self.state.best_val_loss:
                self.state.best_val_loss = val_metrics['val_loss']
                self.save_checkpoint('best_model.pt')
            
            if (epoch + 1) % self.save_every == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt')
        
        self.save_checkpoint('final_model.pt')
        
        return {
            'total_epochs': self.epochs, 'total_time_hours': (time.time() - start_time) / 3600,
            'best_val_loss': self.state.best_val_loss, 'train_losses': self.state.train_losses, 'val_losses': self.state.val_losses
        }


def generate_report(config, model, results, output_dir):
    """生成训练报告"""
    report_dir = Path(output_dir) / 'reports'
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stats = model.get_memory_stats()
    
    report = {'timestamp': timestamp, 'config': config, 'model': stats, 'training': results}
    
    with open(report_dir / f'report_{timestamp}.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    md = f"""# 多模态模型训练报告
**时间**: {timestamp}

## 模型
- 参数量: {stats['total_params']/1e6:.2f}M
- 估计显存: {stats['estimated_training_memory_gb']:.2f}GB

## 训练结果
- 总轮数: {results.get('total_epochs')}
- 训练时间: {results.get('total_time_hours', 0):.2f}小时
- 最佳验证损失: {results.get('best_val_loss', 'N/A'):.4f}

## 输出文件
- 最佳模型: checkpoints/best_model.pt
- 最终模型: checkpoints/final_model.pt
"""
    
    with open(report_dir / f'report_{timestamp}.md', 'w') as f:
        f.write(md)
    
    print(f"报告已保存: {report_dir}")


def main():
    parser = argparse.ArgumentParser(description="多模态模型训练")
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--data-root', type=str, default='./data')
    parser.add_argument('--output-dir', type=str, default='./outputs')
    parser.add_argument('--model-size', type=str, default='base', choices=['tiny', 'small', 'base', 'large'])
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    config = {}
    if Path(args.config).exists() and YAML_AVAILABLE:
        with open(args.config) as f:
            config = yaml.safe_load(f) or {}
    
    config.setdefault('model', {})['size'] = args.model_size
    config.setdefault('training', {}).update({'batch_size': args.batch_size, 'epochs': args.epochs, 'learning_rate': args.lr})
    config.setdefault('paths', {}).update({'data_root': args.data_root, 'output_dir': args.output_dir,
                                           'checkpoint_dir': f'{args.output_dir}/checkpoints', 'log_dir': f'{args.output_dir}/logs'})
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("        多模态交互模型训练系统")
    print("=" * 60)
    
    print("\n1. 创建模型...")
    model = create_model(args.model_size, use_checkpoint=True)
    stats = model.get_memory_stats()
    print(f"   参数量: {stats['total_params']/1e6:.2f}M | 估计显存: {stats['estimated_training_memory_gb']:.2f}GB")
    
    print("\n2. 加载数据...")
    try:
        train_loader, val_loader = create_dataloaders(args.data_root, DataConfig(), batch_size=args.batch_size, num_workers=4)
    except ValueError as e:
        print(f"错误: {e}\n请先运行 download_datasets.py")
        sys.exit(1)
    
    print("\n3. 初始化训练器...")
    trainer = Trainer(model, train_loader, val_loader, config)
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    print("\n4. 开始训练...")
    results = trainer.train()
    
    print("\n5. 生成报告...")
    generate_report(config, model, results, args.output_dir)
    
    print("\n" + "=" * 60)
    print("训练完成!")
    print(f"输出目录: {args.output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
