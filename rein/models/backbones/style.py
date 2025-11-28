from mmseg.models.builder import MODELS
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import reduce
from operator import mul
from torch import Tensor


@MODELS.register_module()
class Style(nn.Module):
    def __init__(
        self,
        num_layers: int,
        embed_dims: int,
        patch_size: int,
        query_dims: int = 256,
        token_length: int = 100,
        use_softmax: bool = True,
        link_token_to_query: bool = True,
        scale_init: float = 0.001,
        zero_mlp_delta_f: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.embed_dims = embed_dims
        self.patch_size = patch_size
        self.query_dims = query_dims
        self.token_length = token_length
        self.link_token_to_query = link_token_to_query
        self.scale_init = scale_init
        self.use_softmax = use_softmax
        self.zero_mlp_delta_f = zero_mlp_delta_f
        self.create_model()

    def create_model(self):
        self.learnable_tokens = nn.Parameter(
            torch.empty([self.num_layers, self.token_length, self.embed_dims])
        )
        self.scale = nn.Parameter(torch.tensor(self.scale_init))
        self.scale_v = nn.Parameter(torch.tensor(self.scale_init))
        self.mlp_token2feat = nn.Linear(self.embed_dims, self.embed_dims)
        self.mlp_delta_f = nn.Linear(self.embed_dims, self.embed_dims)
        self.mlp_token2v = nn.Linear(self.embed_dims, self.embed_dims)
        self.mlp_delta_v = nn.Linear(self.embed_dims, self.embed_dims)
        self.mlp_fusion = nn.Linear(self.embed_dims*2, self.embed_dims)
        val = math.sqrt(
            6.0
            / float(
                3 * reduce(mul, (self.patch_size, self.patch_size), 1) + self.embed_dims
            )
        )
        nn.init.uniform_(self.learnable_tokens.data, -val, val)
        nn.init.kaiming_uniform_(self.mlp_delta_f.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.mlp_token2feat.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.mlp_delta_v.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.mlp_token2v.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.mlp_fusion.weight, a=math.sqrt(5))
        self.transform = nn.Linear(self.embed_dims, self.query_dims)
        self.merge = nn.Linear(self.query_dims * 3, self.query_dims)
        if self.zero_mlp_delta_f:
            del self.scale
            del self.scale_v
            self.scale = 1.0
            self.scale_v = 1.0
            nn.init.zeros_(self.mlp_delta_f.weight)
            nn.init.zeros_(self.mlp_delta_f.bias)
            nn.init.zeros_(self.mlp_delta_v.weight)
            nn.init.zeros_(self.mlp_delta_v.bias)

    def return_auto(self, feats):
        if self.link_token_to_query:
            tokens = self.transform(self.get_tokens(-1)).permute(1, 2, 0)
            tokens = torch.cat(
                [
                    F.max_pool1d(tokens, kernel_size=self.num_layers),
                    F.avg_pool1d(tokens, kernel_size=self.num_layers),
                    tokens[:, :, -1].unsqueeze(-1),
                ],
                dim=-1,
            )
            querys = self.merge(tokens.flatten(-2, -1))
            return feats, querys
        else:
            return feats

    def get_tokens(self, layer: int) -> Tensor:
        if layer == -1:
            # return all
            return self.learnable_tokens
        else:
            return self.learnable_tokens[layer]

    def forward(
        self, feats: Tensor, layer: int, batch_first=False, has_cls_token=True
    ) -> Tensor:
        if batch_first:
            feats = feats.permute(1, 0, 2)
        if has_cls_token:
            cls_token, feats = torch.tensor_split(feats, [1], dim=0)
        tokens = self.get_tokens(layer)
        delta_feat = self.forward_delta_feat(
            feats,
            tokens,
            layer,
        )
        delta_feat = delta_feat * self.scale
        feats = feats + delta_feat
        if has_cls_token:
            feats = torch.cat([cls_token, feats], dim=0)
        if batch_first:
            feats = feats.permute(1, 0, 2)
        return feats

    def forward_delta_feat(self, feats: Tensor, tokens: Tensor, layers: int) -> Tensor:
        attn = torch.einsum("nbc,mc->nbm", feats, tokens)
        if self.use_softmax:
            attn = attn * (self.embed_dims**-0.5)
            attn = F.softmax(attn, dim=-1)
        delta_f = torch.einsum(
            "nbm,mc->nbc",
            attn[:, :, 1:],
            self.mlp_token2feat(tokens[1:, :]),
        )
        delta_f = self.mlp_delta_f(delta_f + feats)
        return delta_f


@MODELS.register_module()
class LoRAStyle(Style):
    def __init__(self, lora_dim=16, **kwargs):
        self.lora_dim = lora_dim
        super().__init__(**kwargs)

    def create_model(self):
        super().create_model()
        del self.learnable_tokens
        self.learnable_tokens_a_feat = nn.Parameter(torch.empty([self.num_layers, self.token_length, self.lora_dim]))
        self.learnable_tokens_b_feat = nn.Parameter(torch.empty([self.num_layers, self.lora_dim, self.embed_dims]))
        val = math.sqrt(6.0 / float(3 * reduce(mul, (self.patch_size, self.patch_size), 1)+ (self.embed_dims * self.lora_dim) ** 0.5))
        nn.init.uniform_(self.learnable_tokens_a_feat.data, -val, val)
        nn.init.uniform_(self.learnable_tokens_b_feat.data, -val, val)

        self.learnable_tokens_a_v = nn.Parameter(torch.empty([self.num_layers, self.token_length, self.lora_dim]))
        self.learnable_tokens_b_v = nn.Parameter(torch.empty([self.num_layers, self.lora_dim, self.embed_dims]))
        val = math.sqrt(6.0 / float(3 * reduce(mul, (self.patch_size, self.patch_size), 1)+ (self.embed_dims * self.lora_dim) ** 0.5))
        nn.init.uniform_(self.learnable_tokens_a_v.data, -val, val)
        nn.init.uniform_(self.learnable_tokens_b_v.data, -val, val)

    def get_tokens(self, layer):
        if layer == -1:
            return self.learnable_tokens_a_feat @ self.learnable_tokens_b_feat
        else:
            return self.learnable_tokens_a_v[layer] @ self.learnable_tokens_b_v[layer], self.learnable_tokens_a_feat[layer] @ self.learnable_tokens_b_feat[layer]
        
    def forward(
        self, feats: Tensor, layer: int, v: Tensor, batch_first=False, has_cls_token=True
    ) -> Tensor:
        if batch_first:
            feats = feats.permute(1, 0, 2)
        if has_cls_token:
            cls_token, feats = torch.tensor_split(feats, [1], dim=0)

        v = v[:, 1:, :, :]
        v = v.view(v.size(0), v.size(1), -1) # (batch, patch=1024, channel=1024)
        v = v.permute(1, 0, 2) # (patch=1024, batch, channel=1024)
        tokens_v, tokens_feat = self.get_tokens(layer)
        delta_v = self.forward_delta_v(
            v,
            tokens_v,
            layer,
        )
        delta_feat = self.forward_delta_feat(
            feats,
            tokens_feat,
            layer,
        )
        fusion = self.mlp_fusion(torch.cat([delta_feat, delta_v], dim=2).permute(1, 0, 2)).permute(1, 0, 2)
        delta_feat = delta_feat * self.scale
        fusion = fusion * self.scale_v
        feats = feats + delta_feat + fusion

        if has_cls_token:
            feats = torch.cat([cls_token, feats], dim=0)
        if batch_first:
            feats = feats.permute(1, 0, 2)
        return feats


    def forward_delta_v(self, v: Tensor, tokens: Tensor, layers: int) -> Tensor:
        attn = torch.einsum("nbc,mc->nbm", v, tokens) # (1024, 4, 1024) @ (100, 1024) = (1024, 4, 100)
        if self.use_softmax:
            attn = attn * (self.embed_dims**-0.5)
            attn = F.softmax(attn, dim=-1)
        delta_v = torch.einsum(
            "nbm,mc->nbc",
            attn,
            self.mlp_token2v(tokens),
        ) # (1024, 4, 100) @ (100, 1024) = (1024, 4, 1024)
        delta_v = self.mlp_delta_v(delta_v + v)
        return delta_v
    
    def forward_delta_feat(self, feats: Tensor, tokens: Tensor, layers: int) -> Tensor:
        attn = torch.einsum("nbc,mc->nbm", feats, tokens)
        if self.use_softmax:
            attn = attn * (self.embed_dims**-0.5)
            attn = F.softmax(attn, dim=-1)
        delta_f = torch.einsum(
            "nbm,mc->nbc",
            attn[:, :, 1:],
            self.mlp_token2feat(tokens[1:, :]),
        )
        delta_f = self.mlp_delta_f(delta_f + feats)
        return delta_f
    
