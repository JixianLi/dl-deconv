"""Plain VDSR-style stack of 3x3 convs (Kim et al. 2016) without the global residual.

VDSR adds its output to the input so the net only learns a correction. That does not
apply here: input (observed) and target (ideal) are normalized separately, so the
input is not a first guess of the target in the same units.
"""

import torch.nn as nn


class DeepCNN(nn.Module):
    def __init__(self, num_layers=8, channels=64, in_channels=1, out_channels=1):
        super().__init__()
        if num_layers < 2:
            raise ValueError(f"num_layers must be >= 2, got {num_layers}")
        self.receptive_field_radius = num_layers  # each 3x3 conv adds 1 px
        layers = [nn.Conv2d(in_channels, channels, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(num_layers - 2):
            layers += [nn.Conv2d(channels, channels, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
        layers.append(nn.Conv2d(channels, out_channels, kernel_size=3, padding=1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
