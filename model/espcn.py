"""ESPCN (Shi et al. 2016), used here at scale=1 as a same-resolution restoration net.

The final conv produces `scale**2` channels per output channel and PixelShuffle
reshuffles them into a `scale`x larger grid. At scale=1 — our observed->ideal
deconvolution setup, where input and target are the same size — PixelShuffle is a
no-op and the network reduces to three convolutions. This is a deliberately small
phase-1 baseline to validate the pipeline; model/deep_cnn.py is the deeper variant.
"""

import torch.nn as nn


class ESPCN(nn.Module):
    receptive_field_radius = 4  # 5x5 + 3x3 + 3x3 convs: 2 + 1 + 1 px

    def __init__(self, scale=1, channels=64, in_channels=1):
        super().__init__()
        mid = channels // 2
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, mid, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.upsample = nn.Sequential(
            nn.Conv2d(mid, in_channels * scale * scale, kernel_size=3, padding=1),
            nn.PixelShuffle(scale),
        )

    def forward(self, x):
        return self.upsample(self.features(x))
