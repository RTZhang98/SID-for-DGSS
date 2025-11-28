# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import os
import warnings

import torch
from torch import Tensor
from torch import nn

import matplotlib.pyplot as plt
import seaborn as sns
import math
from functools import reduce
from operator import mul

logger = logging.getLogger("dinov2")

XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import memory_efficient_attention, unbind

        XFORMERS_AVAILABLE = True
        warnings.warn("xFormers is available (Attention)")
    else:
        warnings.warn("xFormers is disabled (Attention)")
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False
    warnings.warn("xFormers is not available (Attention)")


class Attention(nn.Module):
    def __init__(
        self,
        dim: int = 1024,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class StyleAttention(Attention):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        lora_dim = 16
        token_length = 1024
        embed_dims = 1024
        self.StyleAttn_learnable_tokens_ak = nn.Parameter(torch.empty([token_length, lora_dim]))
        self.StyleAttn_learnable_tokens_bk = nn.Parameter(torch.empty([lora_dim, embed_dims]))
        self.StyleAttn_learnable_tokens_av = nn.Parameter(torch.empty([token_length, lora_dim]))
        self.StyleAttn_learnable_tokens_bv = nn.Parameter(torch.empty([lora_dim, embed_dims]))
        val = math.sqrt(6.0 / float(3 * reduce(mul, (16, 16), 1)+ (embed_dims * lora_dim) ** 0.5))
        nn.init.uniform_(self.StyleAttn_learnable_tokens_ak.data, -val, val)
        nn.init.uniform_(self.StyleAttn_learnable_tokens_bk.data, -val, val)
        nn.init.uniform_(self.StyleAttn_learnable_tokens_av.data, -val, val)
        nn.init.uniform_(self.StyleAttn_learnable_tokens_bv.data, -val, val)

        # Attention 模块，用于生成注意力权重
        self.attentionStyle = nn.Sequential(
            nn.Linear(dim, dim// 4),  # 降维
            nn.ReLU(),
            nn.Linear(dim // 4, 1),  # 恢复维度
            nn.Sigmoid()
        )
        
        # 自适应平均池化，用于 resize GAP 的结果
        #self.resize_pooling = nn.AdaptiveAvgPool2d((output_size, input_channels))

    def get_tokens(self, mode):
        if mode == "k":
            return self.StyleAttn_learnable_tokens_ak @ self.StyleAttn_learnable_tokens_bk
        elif mode == "v":
            return self.StyleAttn_learnable_tokens_av @ self.StyleAttn_learnable_tokens_bv
        else:
            print("style attn mode error!")

    def forward_style(self, x: Tensor, attn_bias=None) -> Tensor:
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x, (k, v)

    def forward(self, x: Tensor, style_k: Tensor, style_v: Tensor, attn_bias=None) -> Tensor:
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)
        B, N, C = x.shape

        # Semantic Preserving step1
        cls_token = x[:,:1,:].clone()
        pc = x[:,1:,:].clone()
        cp = pc.permute(0,2,1)
        attn_weights = self.attentionStyle(cp).squeeze(-1).unsqueeze(1) # (batch, 1024, 1024) -> (batch, 1024, 1024)
        weighted_cls_token = cls_token * attn_weights

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        tokens_k = self.get_tokens("k")
        tokens_v = self.get_tokens("v")
        lora_k = torch.einsum("nbc,mc->nbm", style_k.view(B,N,-1), tokens_k).reshape(B,N,self.num_heads,-1)
        lora_v = torch.einsum("nbc,mc->nbm", style_v.view(B,N,-1), tokens_v).reshape(B,N,self.num_heads,-1)

        x = memory_efficient_attention(q, k + lora_k, v + lora_v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)

        # Semantic Preserving step2
        post_cls_token = x[:,:1,:].clone()
        post_pc = x[:,1:,:].clone()
        x_return = torch.cat([post_cls_token + weighted_cls_token, post_pc], dim=1)

        return x_return, v + lora_v
        # return x, v + lora_v



    def forward_test(self, x: Tensor, attn_bias=None) -> Tensor:
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)
        B, N, C = x.shape

        # Semantic Preserving step1
        cls_token = x[:,:1,:].clone()
        pc = x[:,1:,:].clone()  
        cp = pc.permute(0,2,1)
        attn_weights = self.attentionStyle(cp).squeeze(-1).unsqueeze(1) # (batch, 1024, 1024) -> (batch, 1024, 1024)
        weighted_cls_token = cls_token * attn_weights

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        tokens_k = self.get_tokens("k")
        tokens_v = self.get_tokens("v")
        lora_k = torch.einsum("nbc,mc->nbm", k.view(B,N,-1), tokens_k).reshape(B,N,self.num_heads,-1)
        lora_v = torch.einsum("nbc,mc->nbm", v.view(B,N,-1), tokens_v).reshape(B,N,self.num_heads,-1)

        x = memory_efficient_attention(q, k + lora_k, v + lora_v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        
        # Semantic Preserving step2
        post_cls_token = x[:,:1,:].clone()
        post_pc = x[:,1:,:].clone()
        x_return = torch.cat([post_cls_token + weighted_cls_token, post_pc], dim=1)

        return x_return, v + lora_v
        # return x, v + lora_v
