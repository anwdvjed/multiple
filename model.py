"""
多模态交互模型
model.py

多分支流架构，支持视觉、文本、音频三种模态
显存优化：梯度检查点 + 混合精度 + 分块注意力
"""

import math
from typing import Optional, Dict, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class LayerNorm(nn.Module):
    """LayerNorm with optional bias"""
    def __init__(self, dim: int, eps: float = 1e-6, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, self.eps)


class MemoryEfficientAttention(nn.Module):
    """显存高效的多头注意力"""
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, N, C = x.shape
        
        if context is None:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
        else:
            q = self.qkv(x)[:, :, :C].reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            kv = self.qkv(context)[:, :, C:].reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]

        if hasattr(F, 'scaled_dot_product_attention'):
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            if mask is not None:
                attn = attn.masked_fill(mask == 0, float('-inf'))
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FeedForward(nn.Module):
    """SwiGLU前馈网络"""
    def __init__(self, dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        hidden_dim = hidden_dim or int(dim * 8 / 3)
        hidden_dim = ((hidden_dim + 7) // 8) * 8

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    """通用Transformer块"""
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = MemoryEfficientAttention(dim, num_heads, attn_drop=dropout, proj_drop=dropout)
        self.norm2 = LayerNorm(dim)
        self.mlp = FeedForward(dim, int(dim * mlp_ratio), dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask=mask)
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    """图像Patch嵌入"""
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x).flatten(2).transpose(1, 2)
        return self.norm(x)


class VisionBranch(nn.Module):
    """视觉分支编码器"""
    def __init__(
        self, img_size: int = 224, patch_size: int = 16, embed_dim: int = 512,
        depth: int = 6, num_heads: int = 8, mlp_ratio: float = 4.0,
        dropout: float = 0.1, use_checkpoint: bool = True
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)
        ])
        self.norm = LayerNorm(embed_dim)
        
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = x.shape[0]
        x = self.patch_embed(x)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        for block in self.blocks:
            if self.use_checkpoint and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        
        x = self.norm(x)
        return x[:, 0], x[:, 1:]


class TextBranch(nn.Module):
    """文本分支编码器"""
    def __init__(
        self, vocab_size: int = 32000, max_seq_len: int = 512, embed_dim: int = 512,
        depth: int = 6, num_heads: int = 8, mlp_ratio: float = 4.0,
        dropout: float = 0.1, use_checkpoint: bool = True
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)
        ])
        self.norm = LayerNorm(embed_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N = input_ids.shape
        
        positions = torch.arange(N, device=input_ids.device).unsqueeze(0)
        x = self.token_embed(input_ids) + self.pos_embed(positions)
        x = self.dropout(x)
        
        for block in self.blocks:
            if self.use_checkpoint and self.training:
                x = checkpoint(block, x, attention_mask, use_reentrant=False)
            else:
                x = block(x, attention_mask)
        
        x = self.norm(x)
        
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        else:
            pooled = x.mean(dim=1)
        
        return pooled, x


class AudioBranch(nn.Module):
    """音频分支编码器"""
    def __init__(
        self, input_dim: int = 80, embed_dim: int = 512, depth: int = 4,
        num_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.1,
        use_checkpoint: bool = True
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_dim, embed_dim // 2, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(embed_dim // 2, embed_dim, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
        )
        
        self.pos_embed = nn.Parameter(torch.zeros(1, 5000, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)
        ])
        self.norm = LayerNorm(embed_dim)
        
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.transpose(1, 2)
        x = self.conv_layers(x)
        x = x.transpose(1, 2)
        
        x = x + self.pos_embed[:, :x.size(1)]
        x = self.dropout(x)
        
        for block in self.blocks:
            if self.use_checkpoint and self.training:
                x = checkpoint(block, x, attention_mask, use_reentrant=False)
            else:
                x = block(x, attention_mask)
        
        x = self.norm(x)
        pooled = x.mean(dim=1)
        
        return pooled, x


class CrossModalAttention(nn.Module):
    """跨模态注意力"""
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.norm_q = LayerNorm(dim)
        self.norm_kv = LayerNorm(dim)
        self.attn = MemoryEfficientAttention(dim, num_heads, attn_drop=dropout, proj_drop=dropout)
        self.norm_ff = LayerNorm(dim)
        self.ff = FeedForward(dim, dropout=dropout)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        query = query + self.attn(self.norm_q(query), self.norm_kv(context))
        query = query + self.ff(self.norm_ff(query))
        return query


class CrossModalInteractionBlock(nn.Module):
    """跨模态交互块"""
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1, use_checkpoint: bool = True):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        
        self.v2t_attn = CrossModalAttention(dim, num_heads, dropout)
        self.t2v_attn = CrossModalAttention(dim, num_heads, dropout)
        self.v2a_attn = CrossModalAttention(dim, num_heads, dropout)
        self.a2v_attn = CrossModalAttention(dim, num_heads, dropout)
        self.t2a_attn = CrossModalAttention(dim, num_heads, dropout)
        self.a2t_attn = CrossModalAttention(dim, num_heads, dropout)

    def forward(
        self, vision_feats: Optional[torch.Tensor] = None,
        text_feats: Optional[torch.Tensor] = None,
        audio_feats: Optional[torch.Tensor] = None
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        v_out, t_out, a_out = vision_feats, text_feats, audio_feats
        
        if vision_feats is not None and text_feats is not None:
            if self.use_checkpoint and self.training:
                v_out = checkpoint(self.v2t_attn, v_out, text_feats, use_reentrant=False)
                t_out = checkpoint(self.t2v_attn, t_out, vision_feats, use_reentrant=False)
            else:
                v_out = self.v2t_attn(v_out, text_feats)
                t_out = self.t2v_attn(t_out, vision_feats)
        
        if vision_feats is not None and audio_feats is not None:
            if self.use_checkpoint and self.training:
                v_out = checkpoint(self.v2a_attn, v_out, audio_feats, use_reentrant=False)
                a_out = checkpoint(self.a2v_attn, a_out, vision_feats, use_reentrant=False)
            else:
                v_out = self.v2a_attn(v_out, audio_feats)
                a_out = self.a2v_attn(a_out, vision_feats)
        
        if text_feats is not None and audio_feats is not None:
            if self.use_checkpoint and self.training:
                t_out = checkpoint(self.t2a_attn, t_out, audio_feats, use_reentrant=False)
                a_out = checkpoint(self.a2t_attn, a_out, text_feats, use_reentrant=False)
            else:
                t_out = self.t2a_attn(t_out, audio_feats)
                a_out = self.a2t_attn(a_out, text_feats)
        
        return v_out, t_out, a_out


class MultiModalFusion(nn.Module):
    """门控融合模块"""
    def __init__(self, dim: int, num_modalities: int = 3, dropout: float = 0.1):
        super().__init__()
        self.modal_projs = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_modalities)])
        self.gate = nn.Sequential(
            nn.Linear(dim * num_modalities, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, num_modalities),
            nn.Softmax(dim=-1)
        )
        self.norm = LayerNorm(dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, modality_features: List[torch.Tensor]) -> torch.Tensor:
        projected = [proj(feat) for proj, feat in zip(self.modal_projs, modality_features)]
        concat = torch.cat(projected, dim=-1)
        weights = self.gate(concat)
        stacked = torch.stack(projected, dim=1)
        fused = (stacked * weights.unsqueeze(-1)).sum(dim=1)
        return self.out_proj(self.norm(fused))


class MultiModalInteractionModel(nn.Module):
    """多模态交互模型"""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        
        default_config = {
            'embed_dim': 512, 'num_heads': 8, 'mlp_ratio': 4.0, 'dropout': 0.1,
            'use_checkpoint': True, 'img_size': 224, 'patch_size': 16,
            'vision_depth': 6, 'vocab_size': 32000, 'max_seq_len': 256,
            'text_depth': 6, 'audio_input_dim': 80, 'audio_depth': 4,
            'cross_modal_depth': 2, 'num_classes': 1000,
        }
        
        if config:
            default_config.update(config)
        self.config = default_config
        
        dim = self.config['embed_dim']
        
        # 模态编码器
        self.vision_branch = VisionBranch(
            self.config['img_size'], self.config['patch_size'], dim,
            self.config['vision_depth'], self.config['num_heads'],
            self.config['mlp_ratio'], self.config['dropout'], self.config['use_checkpoint']
        )
        
        self.text_branch = TextBranch(
            self.config['vocab_size'], self.config['max_seq_len'], dim,
            self.config['text_depth'], self.config['num_heads'],
            self.config['mlp_ratio'], self.config['dropout'], self.config['use_checkpoint']
        )
        
        self.audio_branch = AudioBranch(
            self.config['audio_input_dim'], dim, self.config['audio_depth'],
            self.config['num_heads'], self.config['mlp_ratio'],
            self.config['dropout'], self.config['use_checkpoint']
        )
        
        # 跨模态交互
        self.cross_modal_blocks = nn.ModuleList([
            CrossModalInteractionBlock(dim, self.config['num_heads'], self.config['dropout'], self.config['use_checkpoint'])
            for _ in range(self.config['cross_modal_depth'])
        ])
        
        # 融合和输出
        self.fusion = MultiModalFusion(dim, 3, self.config['dropout'])
        self.classifier = nn.Sequential(LayerNorm(dim), nn.Linear(dim, self.config['num_classes']))
        self.contrastive_proj = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 256))
        
        # 双模态投影
        self.dual_proj = nn.Linear(dim * 2, dim)
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, images: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        outputs = {}
        
        vision_pooled, vision_seq = None, None
        text_pooled, text_seq = None, None
        audio_pooled, audio_seq = None, None
        
        if images is not None:
            vision_pooled, vision_seq = self.vision_branch(images)
        if input_ids is not None:
            text_pooled, text_seq = self.text_branch(input_ids, attention_mask)
        if audio is not None:
            audio_pooled, audio_seq = self.audio_branch(audio)
        
        # 跨模态交互
        for block in self.cross_modal_blocks:
            vision_seq, text_seq, audio_seq = block(vision_seq, text_seq, audio_seq)
        
        # 更新池化
        if vision_seq is not None:
            vision_pooled = vision_seq.mean(dim=1)
        if text_seq is not None:
            text_pooled = text_seq.mean(dim=1) if attention_mask is None else \
                (text_seq * attention_mask.unsqueeze(-1).float()).sum(1) / attention_mask.sum(1, keepdim=True).clamp(min=1e-9)
        if audio_seq is not None:
            audio_pooled = audio_seq.mean(dim=1)
        
        # 融合
        modal_features = [f for f in [vision_pooled, text_pooled, audio_pooled] if f is not None]
        
        if len(modal_features) == 0:
            raise ValueError("至少需要一种模态输入")
        elif len(modal_features) == 1:
            fused = modal_features[0]
        elif len(modal_features) == 2:
            fused = self.dual_proj(torch.cat(modal_features, dim=-1))
        else:
            fused = self.fusion(modal_features)
        
        outputs['fused_features'] = fused
        outputs['logits'] = self.classifier(fused)
        outputs['contrastive_embeds'] = F.normalize(self.contrastive_proj(fused), dim=-1)
        
        if return_features:
            outputs.update({
                'vision_features': vision_pooled, 'text_features': text_pooled,
                'audio_features': audio_pooled, 'vision_sequence': vision_seq,
                'text_sequence': text_seq, 'audio_sequence': audio_seq
            })
        
        return outputs

    def get_contrastive_loss(self, embeds1: torch.Tensor, embeds2: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
        logits = embeds1 @ embeds2.T / temperature
        labels = torch.arange(logits.shape[0], device=logits.device)
        return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

    @torch.no_grad()
    def get_memory_stats(self) -> Dict[str, float]:
        total_params = sum(p.numel() for p in self.parameters())
        param_memory_mb = total_params * 4 / (1024 ** 2)
        return {
            'total_params': total_params,
            'param_memory_mb': param_memory_mb,
            'estimated_training_memory_gb': param_memory_mb * 4 / 1024
        }


def create_model(model_size: str = 'base', **kwargs) -> MultiModalInteractionModel:
    """创建不同规模的模型"""
    configs = {
        'tiny': {'embed_dim': 256, 'num_heads': 4, 'vision_depth': 4, 'text_depth': 4, 'audio_depth': 2, 'cross_modal_depth': 1},
        'small': {'embed_dim': 384, 'num_heads': 6, 'vision_depth': 6, 'text_depth': 6, 'audio_depth': 3, 'cross_modal_depth': 2},
        'base': {'embed_dim': 512, 'num_heads': 8, 'vision_depth': 8, 'text_depth': 8, 'audio_depth': 4, 'cross_modal_depth': 2},
        'large': {'embed_dim': 768, 'num_heads': 12, 'vision_depth': 12, 'text_depth': 12, 'audio_depth': 6, 'cross_modal_depth': 3},
    }
    config = configs.get(model_size, configs['base'])
    config.update(kwargs)
    return MultiModalInteractionModel(config)
