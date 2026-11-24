"""A compact U-Net for binary knee-bone segmentation."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class UNet(nn.Module):
    """U-Net that returns one foreground logit per pixel."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_channels: int = 16) -> None:
        super().__init__()
        if min(in_channels, out_channels, base_channels) < 1:
            raise ValueError("Channel counts must be positive")

        channels = [base_channels * (2**level) for level in range(5)]
        self.encoder1 = DoubleConv(in_channels, channels[0])
        self.encoder2 = DoubleConv(channels[0], channels[1])
        self.encoder3 = DoubleConv(channels[1], channels[2])
        self.encoder4 = DoubleConv(channels[2], channels[3])
        self.bottleneck = DoubleConv(channels[3], channels[4])
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.up4 = nn.ConvTranspose2d(channels[4], channels[3], kernel_size=2, stride=2)
        self.decoder4 = DoubleConv(channels[3] * 2, channels[3])
        self.up3 = nn.ConvTranspose2d(channels[3], channels[2], kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(channels[2] * 2, channels[2])
        self.up2 = nn.ConvTranspose2d(channels[2], channels[1], kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(channels[1] * 2, channels[1])
        self.up1 = nn.ConvTranspose2d(channels[1], channels[0], kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(channels[0] * 2, channels[0])
        self.output = nn.Conv2d(channels[0], out_channels, kernel_size=1)

    @staticmethod
    def _join(
        upsampled: torch.Tensor,
        skip: torch.Tensor,
    ) -> torch.Tensor:
        if upsampled.shape[-2:] != skip.shape[-2:]:
            upsampled = functional.interpolate(
                upsampled,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return torch.cat((skip, upsampled), dim=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        enc1 = self.encoder1(inputs)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))
        center = self.bottleneck(self.pool(enc4))

        dec4 = self.decoder4(self._join(self.up4(center), enc4))
        dec3 = self.decoder3(self._join(self.up3(dec4), enc3))
        dec2 = self.decoder2(self._join(self.up2(dec3), enc2))
        dec1 = self.decoder1(self._join(self.up1(dec2), enc1))
        return self.output(dec1)
