from ast import Gt
import numpy as np
from mmcv.cnn import ConvModule
from mmseg.ops import Upsample, resize  # type: ignore

from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from typing import Optional
import math
from functools import partial
from mmcv.runner import auto_fp16, force_fp32  # type: ignore
import matplotlib.pyplot as plt

from timm.models.layers import trunc_normal_
import matplotlib.pyplot as plt
from mmseg.models.losses import accuracy

from models.decode_heads.utils import positional_encoding


def trunc_normal_init(
    module: nn.Module,
    mean: float = 0,
    std: float = 1,
    a: float = -2,
    b: float = 2,
    bias: float = 0,
) -> None:
    if hasattr(module, "weight") and module.weight is not None:
        trunc_normal_(module.weight, mean, std, a, b)  # type: ignore
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)  # type: ignore


def constant_init(module, val, bias=0):
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class TPN_Decoder(TransformerDecoder):
    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ):
        output = tgt
        attns = []
        outputs = []
        for mod in self.layers:
            output, attn = mod(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
            attns.append(attn)
            outputs.append(output)
        if self.norm is not None:  # not do
            output = self.norm(output)

        return outputs, attns


class TPN_DecoderLayer(TransformerDecoderLayer):
    def __init__(self, **kwargs):
        super(TPN_DecoderLayer, self).__init__(**kwargs)
        del self.multihead_attn
        self.multihead_attn = Attention(
            kwargs["d_model"], num_heads=kwargs["nhead"], qkv_bias=True, attn_drop=0.1
        )

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:

        tgt2, attn2 = self.multihead_attn(
            tgt.transpose(0, 1), memory.transpose(0, 1), memory.transpose(0, 1)
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, attn2


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, xq, xk, xv):
        B, Nq, C = xq.size()  # 1, 21, 512
        Nk = xk.size()[1]
        Nv = xv.size()[1]

        q = (
            self.q(xq)
            .reshape(B, Nq, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.k(xk)
            .reshape(B, Nk, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.v(xv)
            .reshape(B, Nv, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_save = attn.clone()
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, Nq, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x.transpose(0, 1), attn_save.sum(dim=1) / self.num_heads


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


@HEADS.register_module()
class ATMSingleHeadSeg(BaseDecodeHead):  # ATM means Attention-based Transfer Module?
    def __init__(
        self,
        img_size,
        in_channels,
        seen_idx,  # antoine: index of the classes that are seen during training
        all_idx,  # antoine: index of all the classes (seen and unseen)
        embed_dims=768,  # antoine: dimension of the CLIP embeddings
        num_layers=3,  # antoine: number of layers in the transformer
        num_heads=8,  # antoine: number of heads in the transformer
        use_stages=1,  # antoine: what is the difference with the number of layers? In the code, it corresponds to the number of layers used for the output of the backbone
        use_proj=True,
        crop_train=False,
        **kwargs,
    ):
        super(ATMSingleHeadSeg, self).__init__(in_channels=in_channels, **kwargs)

        self.image_size = img_size
        self.use_stages = use_stages
        self.crop_train = crop_train
        self.seen_idx = seen_idx
        self.all_idx = all_idx
        nhead = num_heads
        dim = embed_dims
        input_proj = []
        proj_norm = []
        atm_decoders = []

        self.unseen_idx = self.all_idx.copy()
        for i_idx in self.seen_idx:
            self.unseen_idx.remove(i_idx)

        for i in range(self.use_stages):
            # FC layer to change ch
            if use_proj:
                # antoine: if projection, use a linear layer
                proj = nn.Linear(self.in_channels, dim)
                # antoine: input channels = 512, dim = 768, projection to embeddings?
                trunc_normal_(proj.weight, std=0.02)  # antoine: initialize the weights
                # antoine: can speed up the convergence of the training process and can also
                # help avoid issues like vanishing or exploding gradients.
                # The truncated normal distribution is a common choice for weight
                # initialization in many types of neural networks.
            else:
                # antoine: if no projection, use an identity layer = no change
                proj = nn.Identity()
            self.add_module("input_proj_{}".format(i + 1), proj)
            # antoine: add the proj layers to the model
            input_proj.append(proj)
            # norm layer
            if use_proj:
                norm = nn.LayerNorm(dim)
            else:
                norm = nn.Identity()
            self.add_module("proj_norm_{}".format(i + 1), norm)
            # antoine: add the norm layers to the model
            proj_norm.append(norm)
            # decoder layer
            decoder_layer = TPN_DecoderLayer(
                d_model=dim, nhead=nhead, dim_feedforward=dim * 4
            )
            decoder = TPN_Decoder(decoder_layer, num_layers)
            self.add_module("decoder_{}".format(i + 1), decoder)
            # antoine: add the decoder layers to the model
            atm_decoders.append(decoder)

        self.input_proj = input_proj
        self.proj_norm = proj_norm
        self.decoder = atm_decoders

        delattr(self, "conv_seg")

        self.q_proj = nn.Linear(dim * 2, dim)

    def init_weights(self):
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=0.02, bias=0)
                # antoine: initialize the weights of the linear layers with a truncated normal distribution
            elif isinstance(m, nn.LayerNorm):
                constant_init(m, val=1.0, bias=0.0)
                # antoine: initialize the weights of the layer norm layers

    def forward_train(
        self,
        inputs,
        img_metas,
        gt_semantic_seg,
        train_cfg,
        self_training=False,
        st_mask=None,
    ):
        seg_logits = self.forward(inputs)

        if self_training:
            pseudo_semantic_masks = seg_logits["pred_masks"].clone().detach().sigmoid()
            pseudo_semantic_masks[:, self.seen_idx, :, :] = -1
            pseudo_semantic_seg = pseudo_semantic_masks.argmax(dim=1).unsqueeze(1)
            # generate pseudo labels for "transductive" setting
            gt_semantic_seg[gt_semantic_seg == -1] = pseudo_semantic_seg[
                gt_semantic_seg == -1
            ]
            gt_semantic_seg[gt_semantic_seg == -1] = 255
            losses = self.losses(seg_logits, gt_semantic_seg)

        else:
            gt_semantic_seg[gt_semantic_seg == -1] = 255
            losses = self.losses(seg_logits, gt_semantic_seg)

        return losses

    def forward_test(self, inputs, img_metas, test_cfg, self_training):
        return self.forward(inputs, self_training)

    def forward(self, inputs_both, self_training=None):
        inputs = inputs_both[0][0]
        # antoine: inputs will be the output of the backbone
        cls_token = inputs_both[0][1]
        # antoine: cls_token will be the cls output of the backbone
        text_token = inputs_both[1]
        # antoine: text_token will be the output of the text encoder

        x = []
        for stage_ in inputs[: self.use_stages]:
            x.append(self.d4_to_d3(stage_) if stage_.dim() > 3 else stage_)
        x.reverse()
        bs = x[0].size()[0]  # antoine: batch size

        laterals = []  # antoine
        attns = []
        maps_size = []
        qs = []

        for idx, (x_, proj_, norm_) in enumerate(
            zip(x, self.input_proj, self.proj_norm)
        ):
            lateral = norm_(proj_(x_))
            # antoine: apply the projection and normalization layers
            if idx == 0:
                laterals.append(lateral)
                # antoine: if it is the first layer, append the lateral to the list
            else:
                if laterals[idx - 1].size()[1] == lateral.size()[1]:
                    laterals.append(lateral + laterals[idx - 1])
                else:
                    # nearest interpolate
                    l_ = self.d3_to_d4(laterals[idx - 1])
                    l_ = F.interpolate(l_, scale_factor=2, mode="nearest")
                    l_ = self.d4_to_d3(l_)
                    laterals.append(l_ + lateral)

        lateral = laterals[-1]

        # antoine: lateral = images? see the paper with my drawing

        q = self.q_proj(self.get_qs(text_token, cls_token))
        # antoine: get the query with the relationship descriptor
        q = q.transpose(0, 1)

        for idx, decoder_ in enumerate(self.decoder):
            q_, attn_ = decoder_(q, lateral.transpose(0, 1))
            # antoine: apply the decoder
            for q, attn in zip(q_, attn_):
                attn = attn.transpose(-1, -2)
                attn = self.d3_to_d4(attn)
                maps_size.append(attn.size()[-2:])
                qs.append(q.transpose(0, 1))
                attns.append(attn)
        qs = torch.stack(qs, dim=0)

        outputs_seg_masks = []
        size = maps_size[-1]

        for i_attn, attn in enumerate(attns):
            if True:
                outputs_seg_masks.append(
                    F.interpolate(attn, size=size, mode="bilinear", align_corners=False)
                )
            else:
                outputs_seg_masks.append(
                    outputs_seg_masks[i_attn - 1]
                    + F.interpolate(
                        attn, size=size, mode="bilinear", align_corners=False
                    )
                )

        pred = F.interpolate(
            outputs_seg_masks[-1],
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        out = {"pred_masks": pred}

        if self.training:
            outputs_seg_masks = torch.stack(
                outputs_seg_masks, dim=0
            )  # (3, bs, 20, 14, 14)
        else:
            if self_training:
                out["pred"] = self.semantic_inference(
                    out["pred_masks"], self.seen_idx
                )  # (bs, 20, 224, 224)
            else:
                out["pred"] = self.semantic_inference(
                    out["pred_masks"], self.seen_idx, 0.1
                )
            return out["pred"]
        return out

    def semantic_inference(self, mask_pred, seen_idx, weight=0.0):
        mask_pred = mask_pred.sigmoid()
        mask_pred[:, seen_idx] = mask_pred[:, seen_idx] - weight
        return mask_pred

    @torch.jit.unused
    def _set_aux_loss(self, outputs_seg_masks):
        return [
            {"pred_masks": a}
            # for a in zip(outputs_seg_masks[:-1])
            for a in outputs_seg_masks[:-1]
        ]

    def d3_to_d4(self, t):
        # antoine: linear to image
        n, hw, c = t.size()
        if hw % 2 != 0:
            t = t[:, 1:]
        h = w = int(math.sqrt(hw))
        return t.transpose(1, 2).reshape(n, c, h, w)

    def d4_to_d3(self, t):
        # antoine: reverse image to linear
        return t.flatten(-2).transpose(-1, -2)

    def get_qs(self, q, cls):
        # q = [q.cls, q]
        # antoine: relationship descriptor (see paper)
        C, dim = q.shape
        bs, _ = cls.shape
        q = q.expand(bs, -1, -1)
        q1 = torch.einsum("bd,bcd->bcd", cls, q)
        q_ = torch.concat((q1, q), dim=-1)
        return q_

    @force_fp32(apply_to=("seg_logit",))
    def losses(self, seg_logit, seg_label, num_classes=None):
        """Compute segmentation loss."""
        if isinstance(seg_logit, dict):
            # atm loss
            seg_label = seg_label.squeeze(1)

            loss = self.loss_decode(
                seg_logit, seg_label, ignore_index=self.ignore_index
            )

            loss["acc_seg"] = accuracy(
                seg_logit["pred_masks"], seg_label, ignore_index=self.ignore_index
            )
            return loss
