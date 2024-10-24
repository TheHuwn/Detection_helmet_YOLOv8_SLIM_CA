import torch
import torch.nn as nn
import torch.nn.functional as F

# Coordinate Attention block
class CoordinateAttention(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=32):
        super(CoordinateAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels // reduction)

        self.conv_h = nn.Conv2d(in_channels // reduction, out_channels, 1, bias=False)
        self.conv_w = nn.Conv2d(in_channels // reduction, out_channels, 1, bias=False)

    def forward(self, x):
        identity = x
        N, C, H, W = x.size()

        x_h = self.pool_h(x).permute(0, 1, 3, 2)
        x_w = self.pool_w(x)

        out = torch.cat([x_h, x_w], dim=2)
        out = F.relu(self.bn1(self.conv1(out)))

        out_h, out_w = torch.split(out, [H, W], dim=2)
        out_h = torch.sigmoid(self.conv_h(out_h.permute(0, 1, 3, 2)))
        out_w = torch.sigmoid(self.conv_w(out_w))

        out = identity * out_h * out_w
        return out

# SlimNeck block
class SlimNeck(nn.Module):
    def __init__(self, in_channels):
        super(SlimNeck, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU()
        )
        self.ca1 = CoordinateAttention(in_channels // 2, in_channels // 2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, padding=1, groups=in_channels // 2, bias=False),
            nn.Conv2d(in_channels // 2, in_channels // 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU()
        )
        self.ca2 = CoordinateAttention(in_channels // 4, in_channels // 4)

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=3, padding=1, groups=in_channels // 4, bias=False),
            nn.Conv2d(in_channels // 4, in_channels // 8, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU()
        )
        self.ca3 = CoordinateAttention(in_channels // 8, in_channels // 8)

    def forward(self, x):
        x = self.conv1(x)
        x = self.ca1(x)
        x = self.conv2(x)
        x = self.ca2(x)
        x = self.conv3(x)
        x = self.ca3(x)
        return x
