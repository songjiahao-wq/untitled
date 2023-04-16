import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, group_size):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.group_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=group_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # 1. 对输入输入 Residual 进行 AvgPool，1*1 的分组卷积和 Softmax
        residual = x
        avg_pooled = self.avg_pool(residual)
        group_conv = self.group_conv(avg_pooled)
        channel_attention = self.softmax(group_conv)

        # 2. 对输入 Residual、g*g 的分组卷积、Norm、ReLU
        group_conv = nn.Conv2d(residual.shape[1], residual.shape[1], kernel_size=(7, 7), padding=7//2 ,groups=residual.shape[1])(residual)
        norm = nn.BatchNorm2d(residual.shape[1])(group_conv)
        activated = F.relu(norm)

        # 3. 将两个支线的特征进行相乘
        out = channel_attention * activated
        return out

# Example usage:
in_channels = 64
group_size = 8
input_tensor = torch.randn(1, in_channels, 40, 40)  # Batch x Channels x Height x Width
channel_attention_module = ChannelAttention(in_channels, group_size)
output_tensor = channel_attention_module(input_tensor)
print(output_tensor.shape)

# import torch
# import torch.nn as nn
#
#
# class CustomNetwork(nn.Module):
#     def __init__(self, in_channels, out_channels, groups):
#         super(CustomNetwork, self).__init__()
#
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.group_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=groups)
#         self.softmax = nn.Softmax(dim=1)
#
#         self.group_conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), groups=groups)
#         self.norm = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         residual = x
#
#         out1 = self.avgpool(residual)
#         out1 = self.group_conv1(out1)
#         out1 = self.softmax(out1)
#
#         out2 = self.group_conv2(residual)
#         out2 = self.norm(out2)
#         out2 = self.relu(out2)
#
#         out = out1 * out2
#
#         return out
#
#
# # 定义输入和网络参数
# in_channels = 3
# out_channels = 64
# groups = 8
# input_tensor = torch.randn(1, in_channels, 32, 32)
#
# # 创建网络实例并传递输入
# net = CustomNetwork(in_channels, out_channels, groups)
# output = net(input_tensor)
#
# print(output.shape)
