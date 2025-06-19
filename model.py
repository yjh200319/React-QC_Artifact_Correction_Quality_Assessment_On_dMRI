import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConvBlock, self).__init__()
        self.upconv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True)
        )
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.upconv(x1)
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2, diffZ // 2, diffZ - diffZ // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class nnUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(nnUNet, self).__init__()
        self.encoder1 = ConvBlock(in_channels, 32)
        self.encoder2 = ConvBlock(32, 64)
        self.encoder3 = ConvBlock(64, 128)
        # self.encoder4 = ConvBlock(128, 256)
        # self.encoder5 = ConvBlock(256, 512)

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # self.upconv4 = UpConvBlock(512, 256)
        # self.upconv3 = UpConvBlock(256, 128)
        self.upconv2 = UpConvBlock(128, 64)
        self.upconv1 = UpConvBlock(64, 32)

        self.final_conv = nn.Conv3d(32, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        # e4 = self.encoder4(self.pool(e3))
        # e5 = self.encoder5(self.pool(e4))

        # d4 = self.upconv4(e5, e4)
        # d3 = self.upconv3(e4, e3)
        d2 = self.upconv2(e3, e2)
        d1 = self.upconv1(d2, e1)

        out = self.final_conv(d1)
        return out


if __name__ == '__main__':
    # Example usage
    model = nnUNet(in_channels=1, out_channels=1)
    summary(model, input_size=(1, 128, 128, 128), batch_size=2, device='cpu')

    x = torch.randn((1, 1, 128, 128, 128))
    output = model(x)
    print(output.shape)

    """
    600w 参数
    """
