import torch
import torch.nn as nn
import torch.nn.functional as F
class ComplexChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ComplexChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class AvgMaxChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(AvgMaxChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Linear(in_channels * 2, in_channels // reduction_ratio)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_y = self.avg_pool(x).view(b, c)
        max_y = self.max_pool(x).view(b, c)
        y = torch.cat((avg_y, max_y), dim=1)
        # y = torch.add(avg_y, max_y)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
# 使用新的通道注意力模块替换原始模块
class ComplexDualAttentionFusion(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(ComplexDualAttentionFusion, self).__init__()
        self.channel_attention = AvgMaxChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
        self.weight = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x1, x2):
        fusion = x1 + x2
        channel_att = self.channel_attention(fusion)
        spatial_att = self.spatial_attention(fusion)
        attention_weight = torch.sigmoid(self.weight)
        att_fusion = attention_weight * channel_att + (1 - attention_weight) * spatial_att
        out = att_fusion * fusion
        return out

# 示例：
# 假设输入通道数为64
in_channels = 64

# 初始化模型
model = ComplexDualAttentionFusion(in_channels)

# 随机生成输入数据
x1 = torch.randn(1, in_channels, 32, 32)
x2 = torch.randn(1, in_channels, 32, 32)

# 前向传播
output = model(x1, x2)
print(output.shape)
