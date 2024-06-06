import torch.nn as nn
import torch
import numpy as np
from mmseg.registry import MODELS
import patchify


# @MODELS.register_module()
class MultiScales(nn.Module):
    def __init__(self, divisions):
        self.divisions = divisions
        self.original_size = None

    def forward(self, x):
        self.original_size = x.size()
        self.original_type = x.dtype
        out = [x]
        for i in range(self.divisions):
            images = self._create_images(x, self.divisions[i])
            out += images
        return out

    def _create_images(self, x, number_divisions):
        """
        Create images from the original image.

        Args:
            x (torch.Tensor): The original image.
            number_divisions (int): The number of divisions to make in each direction.
                If 2, the image will be divided in 2 parts in each direction, resulting in 4 images.
                The number of images will be number_divisions**2.
        """
        patch_sizes = self.original_size // number_divisions
        x = np.array(x)
        patches = patchify.patchify(x, patch_sizes, step=patch_sizes)
        patches = [x] + patches
        for i in range(len(patches)):
            patches[i] = torch.from_numpy(patches[i])
            patches[i] = patches[i].resize(self.original_size)
        return patches
