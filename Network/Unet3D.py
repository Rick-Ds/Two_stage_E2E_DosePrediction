"""PyTorch implementation of a standard 4-level 3D U-Net for volumetric prediction tasks."""

import torch
import torch.nn as nn

class UNet3D(nn.Module):
    def __init__(
        self,
        in_channels=10,
        out_channels=1,
        base_num_filters=32,
        activation='linear'
    ):
        super().__init__()
        self.activation_name = activation
        self.base_num_filters = base_num_filters


        f1 = base_num_filters
        f2 = base_num_filters * 2
        f3 = base_num_filters * 4
        f4 = base_num_filters * 8
        f5 = base_num_filters * 16


        self.enc1 = self.conv_block(in_channels, f1)
        self.enc2 = self.conv_block(f1, f2)
        self.enc3 = self.conv_block(f2, f3)
        self.enc4 = self.conv_block(f3, f4)

        self.pool = nn.MaxPool3d(2)


        self.bridge = self.conv_block(f4, f5)


        self.up4 = self.up_block(f5, f4)

        self.dec4 = self.conv_block(f4 * 2, f4)

        self.up3 = self.up_block(f4, f3)

        self.dec3 = self.conv_block(f3 * 2, f3)

        self.up2 = self.up_block(f3, f2)

        self.dec2 = self.conv_block(f2 * 2, f2)

        self.up1 = self.up_block(f2, f1)

        self.dec1 = self.conv_block(f1 * 2, f1)

        self.out_conv = nn.Conv3d(f1, out_channels, kernel_size=1)


        if self.activation_name == 'sigmoid':
            self.final_act = nn.Sigmoid()
        elif self.activation_name == 'relu':
            self.final_act = nn.ReLU(inplace=True)
        elif self.activation_name == 'tanh':
            self.final_act = nn.Tanh()
        else:
            self.final_act = nn.Identity()

    @staticmethod
    def conv_block(in_ch, out_ch):
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def up_block(in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):

        e1 = self.enc1(x)
        p1 = self.pool(e1)

        e2 = self.enc2(p1)
        p2 = self.pool(e2)

        e3 = self.enc3(p2)
        p3 = self.pool(e3)

        e4 = self.enc4(p3)
        p4 = self.pool(e4)


        b = self.bridge(p4)


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
    m = UNet3D(in_channels=9, out_channels=1, base_num_filters=16, activation='linear')
    x = torch.randn(1, 9, 64, 256, 256)
    y = m(x)
    print("Output:", y.shape)
