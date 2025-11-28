import os
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T
from mmseg.models.builder import BACKBONES, MODELS
from .style import Style
from .dino_v2_customize import DinoVisionTransformerCustomize
from .utils import set_requires_grad, set_train
import torch.nn.functional as F  

@BACKBONES.register_module()
class StyleDinoVisionTransformer(DinoVisionTransformerCustomize):
    def __init__(
        self,
        style_config=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.style: Style = MODELS.build(style_config)
        self.iter = 0

    def forward_features(self, x, masks=None):
        B, _, h, w = x.shape
        H, W = h // self.patch_size, w // self.patch_size

        if x.shape[1] != 3:
            x=x.view(-1, 3, x.shape[-2], x.shape[-1])
            
            x = self.prepare_tokens_with_masks(x, masks)
            outs = []
            train_x = x[::2] 
            style_x_anchor = x[1::2]   
            content_x_anchor = train_x.clone()

            for idx, blk in enumerate(self.blocks):
                content_x_id = content_x_anchor.clone()
                style_x_id = style_x_anchor.clone()
                with torch.no_grad():
                    content_x_anchor, (content_k, content_v) = blk.forward_style(content_x_anchor)
                    style_x_anchor, (style_k, style_v) = blk.forward_style(style_x_anchor)
                content_x_id, content_v = blk.forward(content_x_id, content_k, content_v)
                style_x_id, style_v = blk.forward(style_x_id, style_k, style_v)

                train_x, v = blk.forward(train_x, style_k, style_v)
                train_x = self.style.forward(
                    train_x,
                    idx,
                    v,
                    batch_first=True,
                    has_cls_token=True,
                )
                if idx in self.out_indices:
                    outs.append([
                        train_x[:, 1:, :].permute(0, 2, 1).reshape(B, -1, H, W).contiguous(),
                        content_x_anchor[:, 1:, :].permute(0, 2, 1).reshape(B, -1, H, W).contiguous(),
                        style_x_anchor[:, 1:, :].permute(0, 2, 1).reshape(B, -1, H, W).contiguous(),
                        content_x_id[:, 1:, :].permute(0, 2, 1).reshape(B, -1, H, W).contiguous(),
                        style_x_id[:, 1:, :].permute(0, 2, 1).reshape(B, -1, H, W).contiguous(),
                    ])
            return self.style.return_auto(outs)
        else:
            x = self.prepare_tokens_with_masks(x, masks)
            outs = []
            for idx, blk in enumerate(self.blocks):
                x, v = blk.forward_test(x)
                x = self.style.forward(
                    x,
                    idx,
                    v,
                    batch_first=True,
                    has_cls_token=True,
                )
                if idx in self.out_indices:
                    outs.append(
                        x[:, 1:, :].permute(0, 2, 1).reshape(B, -1, H, W).contiguous()
                )
            return self.style.return_auto(outs)

    def train(self, mode: bool = True):
        if not mode:
            return super().train(mode)
        set_requires_grad(self, ["style", "attentionStyle", "StyleAttn_learnable_tokens"])
        set_train(self, ["style", "attentionStyle", "StyleAttn_learnable_tokens"])

    def state_dict(self, destination, prefix, keep_vars):
        state = super().state_dict(destination, prefix, keep_vars)
        keys = [k for k in state.keys() if "style" not in k and "attentionStyle" not in k and "StyleAttn_learnable_tokens" not in k]
        for key in keys:
            state.pop(key)
            if key in destination:
                destination.pop(key)
        return state
