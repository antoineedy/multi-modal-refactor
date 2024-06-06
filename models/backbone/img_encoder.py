from collections import OrderedDict
from curses import A_ALTCHARSET
from tkinter import OUTSIDE
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Dropout
from torch import nn
from timm.models.layers import drop, drop_path, trunc_normal_
from mmseg.models.builder import BACKBONES

from mmseg.models.backbones import ResNet
from mmseg.models.backbones import VisionTransformer as MMVisionTransformer

from timm.models.resnet import ResNet as TimmResNet
from timm.models.resnet import Bottleneck as TimmBottleneck

from functools import reduce
from operator import mul

import math
from .utils import *


@BACKBONES.register_module()
class CLIPVisionTransformer(nn.Module):
    def __init__(
        self,
        input_resolution=224,
        patch_size=32,
        width=768,
        layers=12,
        heads=12,
        output_dim=512,
        drop_path_rate=0.0,
        out_indices=[3, 5, 7, 11],
        pretrained=None,
        get_embeddings=False,
        **kwargs,
    ):
        super().__init__()
        self.pretrained = pretrained
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

        scale = width**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width)
        )
        self.spatial_size = input_resolution // patch_size
        self.ln_pre = LayerNorm(width)
        self.get_embeddings = get_embeddings

        self.transformer = Transformer(
            width, layers, heads, drop_path_rate=drop_path_rate
        )

        self.out_indices = out_indices

        if get_embeddings:
            self.ln_post = LayerNorm(width)
            self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        embed_dim = width
        self.patch_size = patch_size

    def init_weights(self, pretrained=None):
        pretrained = pretrained or self.pretrained
        if isinstance(pretrained, str):
            checkpoint = (
                torch.jit.load(pretrained, map_location="cpu").float().state_dict()
            )

            state_dict = {}  # new model

            for k in checkpoint.keys():
                if k.startswith("visual."):
                    new_k = k.replace("visual.", "")
                    state_dict[new_k] = checkpoint[k]

            if "positional_embedding" in state_dict.keys():
                if (
                    self.positional_embedding.shape
                    != state_dict["positional_embedding"].shape
                ):
                    # (1025, 768)                      (197, 768)   upsample the positional_embedding for larger input
                    print(
                        f'Resize the pos_embed shape from {state_dict["positional_embedding"].shape} to {self.positional_embedding.shape}'
                    )
                    cls_pos = state_dict["positional_embedding"][0:1, :]
                    if self.patch_size == 16:
                        spatial_pos = F.interpolate(
                            state_dict["positional_embedding"][1:,]
                            .reshape(1, 14, 14, 768)
                            .permute(0, 3, 1, 2),
                            size=(self.spatial_size, self.spatial_size),
                            mode="bilinear",
                        )
                    elif self.patch_size == 32:
                        spatial_pos = F.interpolate(
                            state_dict["positional_embedding"][1:,]
                            .reshape(1, 7, 7, 768)
                            .permute(0, 3, 1, 2),
                            size=(self.spatial_size, self.spatial_size),
                            mode="bilinear",
                        )
                    else:
                        assert AttributeError("Patch Size should be 16 or 32")
                    spatial_pos = spatial_pos.reshape(
                        768, self.spatial_size * self.spatial_size
                    ).permute(1, 0)
                    positional_embedding = torch.cat([cls_pos, spatial_pos], dim=0)
                    state_dict["positional_embedding"] = positional_embedding
                    assert (
                        self.positional_embedding.shape
                        == state_dict["positional_embedding"].shape
                    )

            u, w = self.load_state_dict(state_dict, False)
            print(
                u, w, "are misaligned params in vision transformer"
            )  # it should be nothing is misaligned

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        B, C, H, W = x.shape
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat(
            [
                self.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                ),
                x,
            ],
            dim=1,
        )

        pos = self.positional_embedding.to(x.dtype)
        cls_pos = pos[0, :] + self.class_embedding.to(x.dtype)
        spatial_pos = F.interpolate(
            pos[1:,]
            .reshape(1, self.spatial_size, self.spatial_size, C)
            .permute(0, 3, 1, 2),
            size=(H, W),
            mode="bilinear",
        )
        spatial_pos = spatial_pos.reshape(1, C, H * W).permute(0, 2, 1)
        pos = torch.cat([cls_pos.reshape(1, 1, C), spatial_pos], dim=1)
        x = x + pos
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND

        features = []
        outs = []
        for i, blk in enumerate(self.transformer.resblocks):
            x = blk(x)
            if len(self.out_indices) > 1:
                if i in self.out_indices:
                    xp = (
                        x.permute(1, 0, 2)[:, 1:, :]
                        .permute(0, 2, 1)
                        .reshape(B, -1, H, W)
                    )
                    features.append(xp.contiguous())

        if self.get_embeddings:
            x = x.permute(1, 0, 2)
            x = self.ln_post(x)
            x = x @ self.proj

            global_embedding = x[:, 0]
            visual_embedding = x[:, 1:].reshape(B, H, W, -1).permute(0, 3, 1, 2)

            if len(self.out_indices) == 1:
                visual_embedding = visual_embedding / visual_embedding.norm(
                    dim=1, keepdim=True
                )
                features.append(visual_embedding)

            outs.append(tuple(features))

            global_embedding = global_embedding / global_embedding.norm(
                dim=1, keepdim=True
            )
            outs.append(global_embedding)

        return outs


@BACKBONES.register_module()
class VPTCLIPVisionTransformer(nn.Module):
    def __init__(
        self,
        input_resolution=224,  # antoine: input_resolution is the size of the input image
        patch_size=32,  # antoine: patch_size is the size of the patch
        width=768,  # antoine: width is the number of features in the patch
        layers=12,  # antoine: layers is the number of layers in the transformer
        heads=12,  # antoine: heads is the number of heads in the transformer
        output_dim=512,  # antoine: ? output dim is not 768?
        drop_path_rate=0.0,
        out_indices=[
            3,
            5,
            7,
            11,
        ],  # antoine: out_indices represents the layers where we want to extract features
        pretrained=None,  # antoine: pretrained is the path to the pretrained model
        get_embeddings=False,  # antoine: get_embeddings is a boolean to know if we want to extract the embeddings
        num_tokens=20,  # antoine: num_tokens is the number of tokens
        prompt_dim=512,  # antoine: prompt_dim is the dimension of the prompt
        total_d_layer=11,  # antoine: total_d_layer is the number of deep layers
        **kwargs,
    ):
        super().__init__()
        self.pretrained = pretrained
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(
            in_channels=3,  # antoine: 3 channels for the input image (RGB)
            out_channels=width,  # antoine: width is the number of features in the patch
            kernel_size=patch_size,
            stride=patch_size,  # antoine: no overlap between patches
            bias=False,
        )
        # antoine: number of parameters in conv1 = 3 * width * patch_size * patch_size = 3 * 768 * 32 * 32

        scale = width**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        # antoine: class_embedding is a learnable parameter of size width
        self.positional_embedding = nn.Parameter(
            scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width)
        )
        # the positional embedding is learnable here?
        self.spatial_size = input_resolution // patch_size
        # antoine: spatial_size is the number of patches in the image
        self.ln_pre = LayerNorm(width)
        self.get_embeddings = get_embeddings
        self.num_layers = layers

        self.transformer = Transformer(
            width, layers, heads, drop_path_rate=drop_path_rate
        )

        self.out_indices = out_indices

        if get_embeddings:
            self.ln_post = LayerNorm(width)
            self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
            # antoine: if we want to extract the embeddings, we need to project the output to the desired dimension

        embed_dim = width

        ## Setting of visual prompt tuning
        self.num_tokens = num_tokens
        self.prompt_dim = prompt_dim
        self.total_d_layer = total_d_layer

        ## Add the prompt parameters # exclude_key=prompt:
        self._init_prompt(
            patch_size, self.num_tokens, self.prompt_dim, self.total_d_layer
        )

    def _init_prompt(self, patch, num_tokens, prompt_dim, total_d_layer):
        patch_size = []
        patch_size.append(patch)
        patch_size.append(patch)
        val = math.sqrt(
            6.0 / float(3 * reduce(mul, patch_size, 1) + prompt_dim)
        )  # noqa

        if total_d_layer >= 0:
            self.prompt_embeddings = nn.Parameter(
                torch.zeros(1, num_tokens, prompt_dim)
            )
            # xavier_uniform initialization
            nn.init.uniform_(self.prompt_embeddings.data, -val, val)

            if total_d_layer > 0:  # noqa
                self.deep_prompt_embeddings = nn.Parameter(
                    torch.zeros(total_d_layer, num_tokens, prompt_dim)
                )
                # xavier_uniform initialization
                nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)

            self.prompt_proj = nn.Linear(prompt_dim, prompt_dim)
            nn.init.kaiming_normal_(self.prompt_proj.weight, a=0, mode="fan_out")
            self.prompt_norm = LayerNorm(prompt_dim, eps=1e-6)
            self.prompt_dropout = Dropout(0.1)

        else:  # total_d_layer < 0
            self.deep_prompt_embeddings = nn.Parameter(
                torch.zeros(abs(total_d_layer), num_tokens, prompt_dim)
            )
            nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)
            self.prompt_proj = nn.Linear(prompt_dim, prompt_dim)
            nn.init.kaiming_normal_(self.prompt_proj.weight, a=0, mode="fan_out")
            self.prompt_norm = LayerNorm(prompt_dim, eps=1e-6)
            self.prompt_dropout = Dropout(0.1)

    def init_weights(self, pretrained=None):
        pretrained = pretrained or self.pretrained
        if isinstance(pretrained, str):
            checkpoint = (
                torch.jit.load(pretrained, map_location="cpu").float().state_dict()
            )

            state_dict = {}

            for k in checkpoint.keys():
                if k.startswith("visual."):
                    new_k = k.replace("visual.", "")
                    state_dict[new_k] = checkpoint[k]

            if "positional_embedding" in state_dict.keys():
                if (
                    self.positional_embedding.shape
                    != state_dict["positional_embedding"].shape
                ):
                    # (1025, 768)                      (197, 768)
                    print(
                        f'Resize the pos_embed shape from {state_dict["positional_embedding"].shape} to {self.positional_embedding.shape}'
                    )
                    cls_pos = state_dict["positional_embedding"][0:1, :]

                    spatial_pos = F.interpolate(
                        state_dict["positional_embedding"][1:,]
                        .reshape(1, 14, 14, 768)
                        .permute(0, 3, 1, 2),
                        size=(self.spatial_size, self.spatial_size),
                        mode="bilinear",
                    )
                    spatial_pos = spatial_pos.reshape(
                        768, self.spatial_size * self.spatial_size
                    ).permute(1, 0)
                    positional_embedding = torch.cat([cls_pos, spatial_pos], dim=0)
                    state_dict["positional_embedding"] = positional_embedding
                    assert (
                        self.positional_embedding.shape
                        == state_dict["positional_embedding"].shape
                    )

            u, w = self.load_state_dict(state_dict, False)
            print(u, w, "are misaligned params in vision transformer")

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # antoine: apply the convolution to the input image.
        B, C, H, W = x.shape
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat(
            [
                self.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                ),
                # antoine: When this zero tensor is added to self.class_embedding, broadcasting occurs.
                # Broadcasting is a PyTorch feature that allows you to perform operations between tensors of different shapes.
                # In this case, self.class_embedding is "stretched" across the second and third dimensions to match the shape
                # of the zero tensor, effectively replicating self.class_embedding for each instance in the batch and each feature in x.
                x,
            ],
            dim=1,
        )

        # antoine: here, we add the class embedding to the tensor, the [cls] token is the first token in the sequence

        pos = self.positional_embedding.to(x.dtype)
        cls_pos = pos[0, :] + self.class_embedding.to(x.dtype)
        spatial_pos = F.interpolate(
            pos[1:,]
            .reshape(1, self.spatial_size, self.spatial_size, C)
            .permute(0, 3, 1, 2),
            size=(H, W),
            mode="bilinear",
        )
        spatial_pos = spatial_pos.reshape(1, C, H * W).permute(0, 2, 1)
        pos = torch.cat([cls_pos.reshape(1, 1, C), spatial_pos], dim=1)
        x = x + pos  # antoine: add the positional embedding to the tensor
        x = self.ln_pre(x)  # antoine: apply layer normalization

        if self.total_d_layer >= 0:
            # concat prompt
            x = torch.cat(
                (
                    x[:, :1, :],
                    self.prompt_dropout(
                        self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1)
                    ),
                    x[:, 1:, :],
                ),
                dim=1,
            )

        x = x.permute(
            1, 0, 2
        )  # antoine: N is the batch size, L is the sequence length, D is the feature dimension

        features = []
        outs = []
        if self.total_d_layer == 0:  # shallow
            for i, blk in enumerate(self.transformer.resblocks):
                x = blk(x)  # antoine: apply the transformer block
                if len(self.out_indices) > 1:
                    # antoine: if we want to extract features from multiple layers
                    if i in self.out_indices:
                        # antoine: if the current layer is in the out_indices
                        xp = (
                            x.permute(1, 0, 2)[:, 1 + self.num_tokens :, :]
                            .permute(0, 2, 1)
                            .reshape(B, -1, H, W)
                        )
                        features.append(xp.contiguous())
                        # antoine: add the features to the list
                        # antoine: contiguous() returns a contiguous tensor containing the same data as self tensor (no worries bro)
        elif self.total_d_layer > 0:  # deep
            x, features = self.forward_deep_prompt(x, features, H, W)
        elif self.total_d_layer < 0:
            x, features = self.forward_reverse_deep_prompt(x, features, H, W)
        else:
            AttributeError("Input correct total_d_layer")

        if self.get_embeddings:
            x = x.permute(1, 0, 2)
            x = self.ln_post(x)
            x = x @ self.proj

            global_embedding = x[:, 0]
            visual_embedding = x[:, -(H * W) :].reshape(B, H, W, -1).permute(0, 3, 1, 2)

            if len(self.out_indices) == 1:
                visual_embedding = visual_embedding / visual_embedding.norm(
                    dim=1, keepdim=True
                )
                features.append(visual_embedding)

            outs.append(tuple(features))
            global_embedding = global_embedding / global_embedding.norm(
                dim=1, keepdim=True
            )
            outs.append(global_embedding)
        return outs

    def forward_deep_prompt(self, embedding_output, features, H, W, out_last=False):
        B = embedding_output.shape[1]

        for i in range(self.num_layers):
            if i == 0:
                hidden_states = self.transformer.resblocks[i](embedding_output)
            elif i <= self.deep_prompt_embeddings.shape[0]:
                deep_prompt_emb = self.prompt_dropout(
                    self.prompt_proj(self.deep_prompt_embeddings[i - 1]).expand(
                        B, -1, -1
                    )
                ).permute(1, 0, 2)
                hidden_states = torch.cat(
                    (
                        hidden_states[:1, :, :],
                        deep_prompt_emb,
                        hidden_states[(1 + self.num_tokens) :, :, :],
                    ),
                    dim=0,
                )

                hidden_states = self.transformer.resblocks[i](hidden_states)
            else:
                hidden_states = torch.cat(
                    (hidden_states[:1, :, :], hidden_states[-(H * W) :, :, :]), dim=0
                )
                hidden_states = self.transformer.resblocks[i](hidden_states)

            if len(self.out_indices) > 1:
                if i in self.out_indices:
                    xp = (
                        hidden_states.permute(1, 0, 2)[:, -(H * W) :, :]
                        .permute(0, 2, 1)
                        .reshape(B, -1, H, W)
                    )
                    features.append(xp.contiguous())

            if i == (self.num_layers - 2):  # 10
                before_last_feats = self.prompt_norm(hidden_states)

        encoded = self.prompt_norm(hidden_states)
        if out_last:
            return before_last_feats
        else:
            return encoded, features

    def forward_reverse_deep_prompt(
        self, embedding_output, features, H, W, out_last=False
    ):
        B = embedding_output.shape[1]
        deep_num_no = (12 - self.deep_prompt_embeddings.shape[0]) - 1

        for i in range(self.num_layers):
            if i == 0:
                hidden_states = self.transformer.resblocks[i](embedding_output)
            elif 0 < i <= deep_num_no:
                hidden_states = self.transformer.resblocks[i](hidden_states)
            else:  ## with deep prompts
                deep_prompt_emb = self.prompt_dropout(
                    self.prompt_proj(
                        self.deep_prompt_embeddings[i - deep_num_no - 1]
                    ).expand(B, -1, -1)
                ).permute(1, 0, 2)
                hidden_states = torch.cat(
                    (
                        hidden_states[:1, :, :],
                        deep_prompt_emb,
                        hidden_states[-(H * W) :, :, :],
                    ),
                    dim=0,
                )

                hidden_states = self.transformer.resblocks[i](hidden_states)

            if len(self.out_indices) > 1:
                if i in self.out_indices:
                    xp = (
                        hidden_states.permute(1, 0, 2)[:, -(H * W) :, :]
                        .permute(0, 2, 1)
                        .reshape(B, -1, H, W)
                    )
                    features.append(xp.contiguous())

            if i == (self.num_layers - 2):
                before_last_feats = self.prompt_norm(hidden_states)

        encoded = self.prompt_norm(hidden_states)
        if out_last:
            return before_last_feats
        else:
            return encoded, features


## TRASH
@BACKBONES.register_module()
class XXScalesVPTCLIPVisionTransformer(nn.Module):
    def __init__(
        self,
        input_resolution=224,
        patch_sizes=[16, 32, 64],  # antoine: changed patch_size to patch_sizes
        width=768,
        layers=12,
        heads=12,
        output_dim=512,
        drop_path_rate=0.0,
        out_indices=[
            3,
            5,
            7,
            11,
        ],
        pretrained=None,
        get_embeddings=False,
        num_tokens=20,
        prompt_dim=512,
        total_d_layer=11,
        **kwargs,
    ):
        super().__init__()
        self.pretrained = pretrained
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1_multiple = [
            nn.Conv2d(
                in_channels=3,
                out_channels=width,
                kernel_size=patch_sizes[i],
                stride=patch_sizes[i],
                bias=False,
            )
            for i in range(len(patch_sizes))
        ]
        # changed to a list of convolutions for the first convolution!

        scale = width**-0.5
        self.class_embeddings = [
            nn.Parameter(scale * torch.randn(width)) for i in range(len(patch_sizes))
        ]
        # antoine: changed to a list of class embeddings!
        self.positional_embeddings = [
            nn.Parameter(
                scale
                * torch.randn((input_resolution // patch_sizes[i]) ** 2 + 1, width)
            )
            for i in range(len(patch_sizes))
        ]
        # changed to a list of positional embeddings!
        self.spatial_sizes = [
            input_resolution // patch_sizes[i] for i in range(len(patch_sizes))
        ]
        # changed to a list of spatial sizes!
        self.ln_pre = LayerNorm(width)
        self.get_embeddings = get_embeddings
        self.num_layers = layers

        self.transformer = Transformer(
            width, layers, heads, drop_path_rate=drop_path_rate
        )

        self.out_indices = out_indices

        if get_embeddings:
            self.ln_post = LayerNorm(width)
            self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
            # antoine: if we want to extract the embeddings, we need to project the output to the desired dimension

        embed_dim = width

        ## Setting of visual prompt tuning
        self.num_tokens = num_tokens
        self.prompt_dim = prompt_dim
        self.total_d_layer = total_d_layer

        ## Add the prompt parameters # exclude_key=prompt:
        self._init_prompts(
            patch_sizes, self.num_tokens, self.prompt_dim, self.total_d_layer
        )

    def _init_prompts(self, patch_sizes, num_tokens, prompt_dim, total_d_layer):
        self.prompt_embeddings_list = []
        self.prompt_proj_list = []
        self.prompt_norm_list = []
        self.prompt_dropout_list = []
        self.deep_prompt_embeddings_list = []
        for i in range(len(patch_sizes)):
            patch = patch_sizes[i]
            patch_size = []
            patch_size.append(patch)
            patch_size.append(patch)
            val = math.sqrt(
                6.0 / float(3 * reduce(mul, patch_size, 1) + prompt_dim)
            )  # noqa

            if total_d_layer >= 0:
                prompt_embeddings = nn.Parameter(torch.zeros(1, num_tokens, prompt_dim))
                # xavier_uniform initialization
                nn.init.uniform_(prompt_embeddings.data, -val, val)

                if total_d_layer > 0:  # noqa
                    deep_prompt_embeddings = nn.Parameter(
                        torch.zeros(total_d_layer, num_tokens, prompt_dim)
                    )
                    # xavier_uniform initialization
                    nn.init.uniform_(deep_prompt_embeddings.data, -val, val)

                prompt_proj = nn.Linear(prompt_dim, prompt_dim)
                nn.init.kaiming_normal_(prompt_proj.weight, a=0, mode="fan_out")
                prompt_norm = LayerNorm(prompt_dim, eps=1e-6)
                prompt_dropout = Dropout(0.1)

            else:  # total_d_layer < 0
                deep_prompt_embeddings = nn.Parameter(
                    torch.zeros(abs(total_d_layer), num_tokens, prompt_dim)
                )
                nn.init.uniform_(deep_prompt_embeddings.data, -val, val)
                prompt_proj = nn.Linear(prompt_dim, prompt_dim)
                nn.init.kaiming_normal_(prompt_proj.weight, a=0, mode="fan_out")
                prompt_norm = LayerNorm(prompt_dim, eps=1e-6)
                prompt_dropout = Dropout(0.1)

            self.prompt_embeddings_list.append(prompt_embeddings)
            self.prompt_proj_list.append(prompt_proj)
            self.prompt_norm_list.append(prompt_norm)
            self.prompt_dropout_list.append(prompt_dropout)
            self.deep_prompt_embeddings_list.append(deep_prompt_embeddings)

    def init_weights(self, pretrained=None):
        pretrained = pretrained or self.pretrained
        if isinstance(pretrained, str):
            checkpoint = (
                torch.jit.load(pretrained, map_location="cpu").float().state_dict()
            )

            state_dict = {}

            for k in checkpoint.keys():
                if k.startswith("visual."):
                    new_k = k.replace("visual.", "")
                    state_dict[new_k] = checkpoint[k]

            if "positional_embedding" in state_dict.keys():
                if (
                    self.positional_embedding.shape
                    != state_dict["positional_embedding"].shape
                ):
                    # (1025, 768)                      (197, 768)
                    print(
                        f'Resize the pos_embed shape from {state_dict["positional_embedding"].shape} to {self.positional_embedding.shape}'
                    )
                    cls_pos = state_dict["positional_embedding"][0:1, :]

                    spatial_pos = F.interpolate(
                        state_dict["positional_embedding"][1:,]
                        .reshape(1, 14, 14, 768)
                        .permute(0, 3, 1, 2),
                        size=(self.spatial_size, self.spatial_size),
                        mode="bilinear",
                    )
                    spatial_pos = spatial_pos.reshape(
                        768, self.spatial_size * self.spatial_size
                    ).permute(1, 0)
                    positional_embedding = torch.cat([cls_pos, spatial_pos], dim=0)
                    state_dict["positional_embedding"] = positional_embedding
                    assert (
                        self.positional_embedding.shape
                        == state_dict["positional_embedding"].shape
                    )

            u, w = self.load_state_dict(state_dict, False)
            print(u, w, "are misaligned params in vision transformer")

    def forward(self, x: torch.Tensor):
        outs_list = []
        for i in range(len(self.conv1_multiple)):
            x = self.conv1_multiple[i](x)
            B, C, H, W = x.shape
            x = x.reshape(x.shape[0], x.shape[1], -1)
            x = x.permute(0, 2, 1)
            x = torch.cat(
                [
                    self.class_embeddings[i].to(x.dtype)
                    + torch.zeros(
                        x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                    ),
                    # antoine: When this zero tensor is added to self.class_embedding, broadcasting occurs.
                    # Broadcasting is a PyTorch feature that allows you to perform operations between tensors of different shapes.
                    # In this case, self.class_embedding is "stretched" across the second and third dimensions to match the shape
                    # of the zero tensor, effectively replicating self.class_embedding for each instance in the batch and each feature in x.
                    x,
                ],
                dim=1,
            )

            # antoine: here, we add the class embedding to the tensor, the [cls] token is the first token in the sequence

            pos = self.positional_embeddings[i].to(x.dtype)
            cls_pos = pos[0, :] + self.class_embeddings[i].to(x.dtype)
            spatial_pos = F.interpolate(
                pos[1:,]
                .reshape(1, self.spatial_sizes[i], self.spatial_sizes[i], C)
                .permute(0, 3, 1, 2),
                size=(H, W),
                mode="bilinear",
            )
            spatial_pos = spatial_pos.reshape(1, C, H * W).permute(0, 2, 1)
            pos = torch.cat([cls_pos.reshape(1, 1, C), spatial_pos], dim=1)
            x = x + pos  # antoine: add the positional embedding to the tensor
            x = self.ln_pre(x)  # antoine: apply layer normalization

            if self.total_d_layer >= 0:
                # concat prompt
                x = torch.cat(
                    (
                        x[:, :1, :],
                        self.prompt_dropout_list[i](
                            self.prompt_proj_list[i](
                                self.prompt_embeddings_list[i]
                            ).expand(B, -1, -1)
                        ),
                        x[:, 1:, :],
                    ),
                    dim=1,
                )

            x = x.permute(
                1, 0, 2
            )  # antoine: N is the batch size, L is the sequence length, D is the feature dimension

            features = []
            outs = []
            if self.total_d_layer == 0:  # shallow
                for i, blk in enumerate(self.transformer.resblocks):
                    x = blk(x)  # antoine: apply the transformer block
                    if len(self.out_indices) > 1:
                        # antoine: if we want to extract features from multiple layers
                        if i in self.out_indices:
                            # antoine: if the current layer is in the out_indices
                            xp = (
                                x.permute(1, 0, 2)[:, 1 + self.num_tokens :, :]
                                .permute(0, 2, 1)
                                .reshape(B, -1, H, W)
                            )
                            features.append(xp.contiguous())
                            # antoine: add the features to the list
                            # antoine: contiguous() returns a contiguous tensor containing the same data as self tensor (no worries bro)
            elif self.total_d_layer > 0:  # deep
                x, features = self.forward_deep_prompt(x, features, H, W, i)
            elif self.total_d_layer < 0:
                x, features = self.forward_reverse_deep_prompt(x, features, H, W, i)
            else:
                AttributeError("Input correct total_d_layer")

            if self.get_embeddings:
                x = x.permute(1, 0, 2)
                x = self.ln_post(x)
                x = x @ self.proj

                global_embedding = x[:, 0]
                visual_embedding = (
                    x[:, -(H * W) :].reshape(B, H, W, -1).permute(0, 3, 1, 2)
                )

                if len(self.out_indices) == 1:
                    visual_embedding = visual_embedding / visual_embedding.norm(
                        dim=1, keepdim=True
                    )
                    features.append(visual_embedding)

                outs.append(tuple(features))
                global_embedding = global_embedding / global_embedding.norm(
                    dim=1, keepdim=True
                )
                outs.append(global_embedding)
            outs_list.append(outs)
        return outs_list

    def forward_deep_prompt(
        self, embedding_output, features, H, W, index, out_last=False
    ):
        # added index to the function signature because of the multi-scale implementation
        B = embedding_output.shape[1]

        deep_prompt_embeddings = self.deep_prompt_embeddings_list[index]
        prompt_dropout = self.prompt_dropout_list[index]
        prompt_proj = self.prompt_proj_list[index]
        prompt_norm = self.prompt_norm_list[index]

        for i in range(self.num_layers):
            if i == 0:
                hidden_states = self.transformer.resblocks[i](embedding_output)
            elif i <= deep_prompt_embeddings.shape[0]:
                deep_prompt_emb = prompt_dropout(
                    prompt_proj(deep_prompt_embeddings[i - 1]).expand(B, -1, -1)
                ).permute(1, 0, 2)
                hidden_states = torch.cat(
                    (
                        hidden_states[:1, :, :],
                        deep_prompt_emb,
                        hidden_states[(1 + self.num_tokens) :, :, :],
                    ),
                    dim=0,
                )

                hidden_states = self.transformer.resblocks[i](hidden_states)
            else:
                hidden_states = torch.cat(
                    (hidden_states[:1, :, :], hidden_states[-(H * W) :, :, :]), dim=0
                )
                hidden_states = self.transformer.resblocks[i](hidden_states)

            if len(self.out_indices) > 1:
                if i in self.out_indices:
                    xp = (
                        hidden_states.permute(1, 0, 2)[:, -(H * W) :, :]
                        .permute(0, 2, 1)
                        .reshape(B, -1, H, W)
                    )
                    features.append(xp.contiguous())

            if i == (self.num_layers - 2):  # 10
                before_last_feats = prompt_norm(hidden_states)

        encoded = prompt_norm(hidden_states)
        if out_last:
            return before_last_feats
        else:
            return encoded, features

    def forward_reverse_deep_prompt(
        self, embedding_output, features, H, W, index, out_last=False
    ):
        B = embedding_output.shape[1]
        deep_prompt_embeddings = self.deep_prompt_embeddings_list[index]
        prompt_dropout = self.prompt_dropout_list[index]
        prompt_proj = self.prompt_proj_list[index]
        prompt_norm = self.prompt_norm_list[index]

        deep_num_no = (12 - deep_prompt_embeddings.shape[0]) - 1

        for i in range(self.num_layers):
            if i == 0:
                hidden_states = self.transformer.resblocks[i](embedding_output)
            elif 0 < i <= deep_num_no:
                hidden_states = self.transformer.resblocks[i](hidden_states)
            else:  ## with deep prompts
                deep_prompt_emb = prompt_dropout(
                    prompt_proj(deep_prompt_embeddings[i - deep_num_no - 1]).expand(
                        B, -1, -1
                    )
                ).permute(1, 0, 2)
                hidden_states = torch.cat(
                    (
                        hidden_states[:1, :, :],
                        deep_prompt_emb,
                        hidden_states[-(H * W) :, :, :],
                    ),
                    dim=0,
                )

                hidden_states = self.transformer.resblocks[i](hidden_states)

            if len(self.out_indices) > 1:
                if i in self.out_indices:
                    xp = (
                        hidden_states.permute(1, 0, 2)[:, -(H * W) :, :]
                        .permute(0, 2, 1)
                        .reshape(B, -1, H, W)
                    )
                    features.append(xp.contiguous())

            if i == (self.num_layers - 2):
                before_last_feats = prompt_norm(hidden_states)

        encoded = prompt_norm(hidden_states)
        if out_last:
            return before_last_feats
        else:
            return encoded, features
