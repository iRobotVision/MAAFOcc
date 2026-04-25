from typing import List, Tuple

import torch
from torch import nn

from mmdet3d.models.builder import FUSERS

import torch.nn.functional as F


@FUSERS.register_module()
class LidarConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
       
        self.out_conv = nn.Sequential(nn.Conv2d(in_channels[1], out_channels, 3, 1, 1, bias=False),
                                      nn.BatchNorm2d(out_channels),
                                      nn.ReLU(inplace=True))


    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        out = self.out_conv(x[0])
        return out

      