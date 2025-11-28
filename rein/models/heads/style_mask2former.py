from mmseg.models.decode_heads.mask2former_head import Mask2FormerHead
from mmseg.registry import MODELS
from mmseg.utils import SampleList
from torch import Tensor
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.models.builder import MODELS
from mmseg.utils import ConfigType, SampleList


@MODELS.register_module()
class StyleMask2FormerHead(Mask2FormerHead):
    def __init__(self, replace_query_feat=False, threshold=0.3, **kwargs):
        super().__init__(**kwargs)
        feat_channels = kwargs["feat_channels"]
        del self.query_embed
        self.vpt_transforms = nn.ModuleList()
        self.replace_query_feat = replace_query_feat
        self.threshold = threshold
        if replace_query_feat:
            del self.query_feat
            self.querys2feat = nn.Linear(feat_channels, feat_channels)
            
    def forward(
        self, x: Tuple[List[Tensor], List[Tensor]], batch_data_samples: SampleList
    ) -> Tuple[List[Tensor]]:
        x, query_embed = x
        if isinstance(x[0], list):
            x_output = [layer[0] for layer in x] 
        else:
            x_output = x
        batch_img_metas = [data_sample.metainfo for data_sample in batch_data_samples]
        batch_size = len(batch_img_metas)
        if query_embed.ndim == 2:
            query_embed = query_embed.expand(batch_size, -1, -1)
        # use vpt_querys to replace query_embed
        mask_features, multi_scale_memorys = self.pixel_decoder(x_output)
        # multi_scale_memorys (from low resolution to high resolution)
        decoder_inputs = []
        decoder_positional_encodings = []
        for i in range(self.num_transformer_feat_level):
            decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i])
            # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
            decoder_input = decoder_input.flatten(2).permute(0, 2, 1)
            level_embed = self.level_embed.weight[i].view(1, 1, -1)
            decoder_input = decoder_input + level_embed
            # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
            mask = decoder_input.new_zeros(
                (batch_size,) + multi_scale_memorys[i].shape[-2:], dtype=torch.bool
            )
            decoder_positional_encoding = self.decoder_positional_encoding(mask)
            decoder_positional_encoding = decoder_positional_encoding.flatten(
                2
            ).permute(0, 2, 1)
            decoder_inputs.append(decoder_input)
            decoder_positional_encodings.append(decoder_positional_encoding)
        # shape (num_queries, c) -> (batch_size, num_queries, c)
        if self.replace_query_feat:
            query_feat = self.querys2feat(query_embed)
        else:
            query_feat = self.query_feat.weight.unsqueeze(0).repeat((batch_size, 1, 1))

        # query_embed = self.query_embed.weight.unsqueeze(0).repeat((batch_size, 1, 1))

        cls_pred_list = []
        mask_pred_list = []
        cls_pred, mask_pred, attn_mask = self._forward_head(
            query_feat, mask_features, multi_scale_memorys[0].shape[-2:]
        )
        cls_pred_list.append(cls_pred)
        mask_pred_list.append(mask_pred)

        for i in range(self.num_transformer_decoder_layers):
            level_idx = i % self.num_transformer_feat_level
            # if a mask is all True(all background), then set it all False.
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            # cross_attn + self_attn
            layer = self.transformer_decoder.layers[i]
            query_feat = layer(
                query=query_feat,
                key=decoder_inputs[level_idx],
                value=decoder_inputs[level_idx],
                query_pos=query_embed,
                key_pos=decoder_positional_encodings[level_idx],
                cross_attn_mask=attn_mask,
                query_key_padding_mask=None,
                # here we do not apply masking on padded region
                key_padding_mask=None,
            )
            cls_pred, mask_pred, attn_mask = self._forward_head(
                query_feat,
                mask_features,
                multi_scale_memorys[(i + 1) % self.num_transformer_feat_level].shape[
                    -2:
                ],
            )

            cls_pred_list.append(cls_pred)
            mask_pred_list.append(mask_pred)

        return cls_pred_list, mask_pred_list

    def loss(self, x: Tuple[Tensor], batch_data_samples: SampleList, train_cfg: ConfigType) -> dict:
        """Perform forward propagation and loss calculation of the decoder head
        on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Multi-level features from the upstream
                network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`SegDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_sem_seg`.
            train_cfg (ConfigType): Training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components.
        """
        # batch SegDataSample to InstanceDataSample
        batch_gt_instances, batch_img_metas = self._seg_data_to_instance_data(
            batch_data_samples)

        # forward
        all_cls_scores, all_mask_preds = self(x, batch_data_samples)

        # loss
        losses = self.loss_by_feat(all_cls_scores, all_mask_preds,
                                   batch_gt_instances, batch_img_metas)
        
        # Style losses
        style_loss = 0.0
        content_loss = 0.0
        id_loss = 0.0

        for layer_idx, features in enumerate(x[0][:-1]):  # `x` is a list of lists
            # Unpack the features
            output, content_anchor, style_anchor, content_id, style_id = features  # Each `features` contains (output, content, style)

            # Step 3: Compute normalization for output and content
            # Normalize output and content
            def normalize(tensor):
                mean = tensor.mean(dim=(2, 3), keepdim=True)  # Compute mean along spatial dimensions
                std = tensor.std(dim=(2, 3), keepdim=True)    # Compute std along spatial dimensions
                return (tensor - mean) / (std + 1e-8)

            output_norm = normalize(output)
            content_norm = normalize(content_anchor)

            # Compute MSE loss between normalized output and content
            layer_content_loss = F.mse_loss(output_norm, content_norm)
            content_loss += layer_content_loss * 10

            # Step 4: Compute mean and variance for output and style
            def compute_mean_variance(tensor):
                mean = tensor.mean(dim=(2, 3), keepdim=True)  # Mean along spatial dimensions
                variance = tensor.var(dim=(2, 3), keepdim=True)  # Variance along spatial dimensions
                return mean, variance

            output_mean, output_var = compute_mean_variance(output)
            style_mean, style_var = compute_mean_variance(style_anchor)

            # Compute MSE loss between output and style (mean and variance)
            layer_style_loss = F.mse_loss(output_mean, style_mean) + F.mse_loss(output_var, style_var)
            style_loss += layer_style_loss * 10

            content_id_norm = normalize(content_id)
            style_id_norm = normalize(style_id)
            content_anchor_norm = normalize(content_anchor)
            style_anchor_norm = normalize(style_anchor)
            layer_id_loss = F.mse_loss(content_id_norm, content_anchor_norm) + F.mse_loss(style_id_norm, style_anchor_norm)
            id_loss += layer_id_loss * 5
            # Optionally, log layer-specific losses for debugging
            losses[f"layer_{layer_idx}_content_loss"] = layer_content_loss
            losses[f"layer_{layer_idx}_style_loss"] = layer_style_loss
            losses[f"layer_{layer_idx}_id_loss"] = layer_id_loss

        # Step 5: Combine losses
        losses["content_loss"] = content_loss
        losses["style_loss"] = style_loss
        losses["id_loss"] = id_loss

        return losses