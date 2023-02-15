import math
import numpy as np
import torch
from torch import nn


class PositionEncode1D(nn.Module):
    def __init__(self, dim, max_length):
        super().__init__()
        assert (dim % 2 == 0)
        self.max_length = max_length

        d = torch.exp(torch.arange(0., dim, 2) * (-math.log(10000.0) / dim))
        position = torch.arange(0., max_length).unsqueeze(1)
        pos = torch.zeros(1, max_length, dim)
        pos[0, :, 0::2] = torch.sin(position * d)
        pos[0, :, 1::2] = torch.cos(position * d)
        self.register_buffer('pos', pos)

    def forward(self, x):
        batch_size, T, dim = x.shape
        x = x + self.pos[:, :T]
        return x


# https://github.com/wzlxjtu/PositionalEncoding2D/blob/master/positionalembedding2d.py
class PositionEncode2D(nn.Module):
    def __init__(self, dim, width, height):
        super().__init__()
        assert (dim % 4 == 0)
        self.width = width
        self.height = height

        dim = dim // 2
        d = torch.exp(torch.arange(0., dim, 2) * -(math.log(10000.0) / dim))
        position_w = torch.arange(0., width).unsqueeze(1)
        position_h = torch.arange(0., height).unsqueeze(1)
        pos = torch.zeros(1, dim * 2, height, width)

        pos[0, 0:dim:2, :, :] = torch.sin(position_w * d).transpose(0, 1).unsqueeze(1).repeat(1, 1, height, 1)
        pos[0, 1:dim:2, :, :] = torch.cos(position_w * d).transpose(0, 1).unsqueeze(1).repeat(1, 1, height, 1)
        pos[0, dim + 0::2, :, :] = torch.sin(position_h * d).transpose(0, 1).unsqueeze(2).repeat(1, 1, 1, width)
        pos[0, dim + 1::2, :, :] = torch.cos(position_h * d).transpose(0, 1).unsqueeze(2).repeat(1, 1, 1, width)
        self.register_buffer('pos', pos)

    def forward(self, x):
        batch_size, C, H, W = x.shape
        x = x + self.pos[:, :, :H, :W]
        return x


def decoder_attention_mask(input_ids: torch.Tensor) -> torch.Tensor:
    batch_size, length = input_ids.shape
    mask_np = 1 - np.triu(np.ones((length, length)), k=1).astype('uint8')
    mask = torch.autograd.Variable(torch.from_numpy(mask_np)).type_as(input_ids)
    return mask.type(torch.DoubleTensor)
