import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.fft


class SpectralAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpectralAttention, self).__init__()

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // 16)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // 16, in_channels)

    def forward(self, x):
        b, c, h, w = x.size()

        # 计算注意力权重
        y = self.avgpool(x).view(b, c)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        attention_weights = torch.sigmoid(y).view(b, c, 1, 1)

        # 计算傅里叶变换
        x_fft = torch.fft.fft2(x)

        # 应用注意力权重
        x_fft_weighted = x_fft * attention_weights

        # 反傅里叶变换
        x_weighted = torch.fft.ifft2(x_fft_weighted).real

        return x_weighted


# 示例
input_tensor = torch.randn(1, 64, 32, 32)
spectral_attention = SpectralAttention(64)
output = spectral_attention(input_tensor)

print(output.shape)

import torch
import torch.nn as nn


class MultiScaleFrequencyAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, pool_sizes=[1, 2, 4]):
        super().__init__()

        # 定义卷积层和池化层
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // reduction_ratio, kernel_size=1)
        self.pools = nn.ModuleList([nn.AdaptiveAvgPool2d(s) for s in pool_sizes])
        self.fcs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_channels // reduction_ratio, in_channels),
                nn.Sigmoid()
            ) for _ in pool_sizes
        ])

    def forward(self, x):
        # 计算频域特征向量
        x_fft = torch.rfft(x, signal_ndim=2)
        x_abs = (x_fft ** 2).sum(dim=-1).sqrt()

        # 计算每个尺度下的注意力权重
        attention_weights = []
        for pool, fc in zip(self.pools, self.fcs):
            y = pool(x_abs)
            y = torch.flatten(y, start_dim=1)
            y = fc(y)
            attention_weights.append(y)

        # 将不同尺度下的注意力权重相加
        attention_weight = sum(attention_weights)
        attention_weight = attention_weight.unsqueeze(-1).unsqueeze(-1)

        # 将注意力权重应用到原始信号上
        x = x_abs * attention_weight
        y = torch.irfft(x, signal_ndim=2, signal_sizes=x_fft.size()[2:])

        return y
# 轻量化的多尺度空间注意力模块
import torch
import torch.nn as nn
import torch.nn.functional as F

class LightweightMultiScaleSpatialAttention(nn.Module):
    def __init__(self, in_channels, kernel_sizes=[3, 5, 7], reduction_ratio=16):
        super(LightweightMultiScaleSpatialAttention, self).__init__()

        self.branches = nn.ModuleList()
        mid_channels = in_channels // reduction_ratio

        for kernel_size in kernel_sizes:
            padding = kernel_size // 2
            branch = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, mid_channels, kernel_size=kernel_size, stride=1, padding=padding, groups=mid_channels),
                nn.Conv2d(mid_channels, in_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            )
            self.branches.append(branch)

    def forward(self, x):
        # x: input tensor of shape (batch_size, in_channels, height, width)

        # Apply different kernel sizes
        attention_maps = []
        for branch in self.branches:
            attention_map = branch(x)
            attention_map = F.sigmoid(attention_map)
            attention_maps.append(attention_map)

        # Element-wise sum of attention maps
        combined_attention_map = torch.sum(torch.stack(attention_maps), dim=0)

        # Apply the combined attention map to the input
        out = x * combined_attention_map

        return out
