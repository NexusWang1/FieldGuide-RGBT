# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""CBAM (Woo et al., ECCV 2018) — 通道+空间注意力，注意力路线对照臂专用。

自写实现，不依赖 ultralytics 内置注意力模块，保证与论文引用口径一致：
    Channel: avg-pool + max-pool -> 共享 MLP(r=16) -> sigmoid
    Spatial: channel-avg + channel-max -> 7x7 conv -> sigmoid
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, c: int, r: int = 16):
        super().__init__()
        h = max(1, c // r)
        self.mlp = nn.Sequential(
            nn.Linear(c, h), nn.ReLU(inplace=True), nn.Linear(h, c)
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        s = torch.sigmoid(self.mlp(x.mean((2, 3))) + self.mlp(x.amax((2, 3))))
        return x * s.view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    def __init__(self, k: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, k, padding=k // 2, bias=False)

    def forward(self, x):
        s = torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)
        return x * torch.sigmoid(self.conv(s))


class CBAM(nn.Module):
    """f' = SA(CA(f))，残差外不做（与原始论文一致，纯乘性调制）。"""

    def __init__(self, c: int, r: int = 16, k: int = 7):
        super().__init__()
        self.ca = ChannelAttention(c, r)
        self.sa = SpatialAttention(k)

    def forward(self, x):
        return self.sa(self.ca(x))
