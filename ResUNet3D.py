"""PyTorch implementation of a 3D Res-UNet with residual blocks for volumetric prediction tasks."""

import torch
import torch.nn as nn


class ResidualBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm3d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm3d(out_ch)

        self.proj = None
        if in_ch != out_ch:
            self.proj = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm3d(out_ch)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.proj is not None:
            identity = self.proj(identity)

        out = out + identity
        out = self.relu(out)
        return out


class ResUNet3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_num_filters: int = 64,
        activation: str = 'linear'
    ):
        super().__init__()

        base = base_num_filters


        self.enc1 = ResidualBlock3D(in_channels, base)
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = ResidualBlock3D(base, base * 2)
        self.pool2 = nn.MaxPool3d(2)

        self.enc3 = ResidualBlock3D(base * 2, base * 4)
        self.pool3 = nn.MaxPool3d(2)

        self.enc4 = ResidualBlock3D(base * 4, base * 8)
        self.pool4 = nn.MaxPool3d(2)


        self.bridge = ResidualBlock3D(base * 8, base * 16)


        self.up4  = nn.ConvTranspose3d(base * 16, base * 8, kernel_size=2, stride=2)
        self.dec4 = ResidualBlock3D(base * 8 + base * 8, base * 8)

        self.up3  = nn.ConvTranspose3d(base * 8, base * 4, kernel_size=2, stride=2)
        self.dec3 = ResidualBlock3D(base * 4 + base * 4, base * 4)

        self.up2  = nn.ConvTranspose3d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = ResidualBlock3D(base * 2 + base * 2, base * 2)

        self.up1  = nn.ConvTranspose3d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = ResidualBlock3D(base + base, base)


        self.out_conv = nn.Conv3d(base, out_channels, kernel_size=1)

        if activation == 'linear':
            self.final_act = nn.Identity()
        elif activation == 'sigmoid':
            self.final_act = nn.Sigmoid()
        elif activation == 'relu':
            self.final_act = nn.ReLU(inplace=True)
        elif activation == 'tanh':
            self.final_act = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x):

        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        e4 = self.enc4(p3)
        p4 = self.pool4(e4)


        b  = self.bridge(p4)


        u4 = self.up4(b)
        d4 = self.dec4(torch.cat([u4, e4], dim=1))

        u3 = self.up3(d4)
        d3 = self.dec3(torch.cat([u3, e3], dim=1))

        u2 = self.up2(d3)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        out = self.out_conv(d1)
        return self.final_act(out)


if __name__ == "__main__":

    m = ResUNet3D(in_channels=10, out_channels=1,
                  base_num_filters=32, activation='linear')
    x = torch.randn(1, 10, 64, 256, 256)
    y = m(x)
    print("Output:", y.shape)
