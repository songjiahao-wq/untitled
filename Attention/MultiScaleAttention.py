import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import torchvision.transforms as transforms
"""
#num_scales参数决定了注意力机制将考虑的不同尺度的数量。这个值取决于你想在你的模型中捕获的多尺度背景的水平和输入图像的大小。
#没有一个放之四海而皆准的数值，因为它取决于你正在处理的具体问题和数据集。
#对于CIFAR-10数据集，它有32x32的图像，你可以从num_scales的值开始，等于3。
#这将允许注意力机制使用扩张的卷积在三种不同的感受野尺寸（3x3，5x5和7x7）上捕捉上下文。
"""
class SpatialMultiScaleAttention(nn.Module):
    def __init__(self, in_channels, num_scales):
        super(SpatialMultiScaleAttention, self).__init__()
        self.in_channels = in_channels
        self.num_scales = num_scales

        # Define the attention layers
        self.attention_layers = nn.ModuleList()
        for i in range(num_scales):
            self.attention_layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=i+1, dilation=i+1))

    def forward(self, x):
        # x: input tensor of shape (batch_size, in_channels, height, width)
        batch_size, _, height, width = x.size()

        # Calculate attention maps for each scale
        attention_maps = []
        for i in range(self.num_scales):
            attention_map = self.attention_layers[i](x)
            attention_map = F.softmax(attention_map.view(batch_size, -1), dim=1)
            attention_map = attention_map.view(batch_size, self.in_channels, height, width)
            attention_maps.append(attention_map)

        # Sum attention maps
        attention_maps = sum(attention_maps)

        # Apply attention to the input
        x = x * attention_maps
        return x
class ChannelAttentionModified(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttentionModified, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Linear(in_channels, in_channels // reduction_ratio, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_channels / reduction_ratio), in_channels),
        )
    def forward(self, x):
        batch_size, channels, _, _ = x.size()

        avg_pool = self.avg_pool(x).view(batch_size, channels)
        max_pool = self.max_pool(x).view(batch_size, channels)

        channel_att_avg = self.relu(self.fc(avg_pool)).view(batch_size, channels, 1, 1)
        channel_att_max = self.relu(self.fc(max_pool)).view(batch_size, channels, 1, 1)
        # channel_att_avg = self.relu(self.fc(avg_pool))
        # channel_att_max = self.relu(self.fc(max_pool))
        #
        # if batch_size > 1:
        #     y = channel_att_avg.view(batch_size, channels, 1, 1)
        #     channel_att_avg = channel_att_avg.view(batch_size, channels, 1, 1)
        #     channel_att_max = channel_att_max.view(batch_size, channels, 1, 1)
        # else:
        #     channel_att_avg = channel_att_avg.view(channels, 1, 1)
        #     channel_att_max = channel_att_max.view(channels, 1, 1)
        channel_att = channel_att_avg + channel_att_max
        return x * channel_att

class MultiScaleAttention(nn.Module):
    def __init__(self, in_channels, out_channels,  num_scales):
        super(MultiScaleAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.spatial_attention = SpatialMultiScaleAttention(in_channels, num_scales)
        self.channel_attention = ChannelAttentionModified(in_channels)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        # x = F.relu(self.conv1(x))

        weight = self.spatial_attention(x) * self.channel_attention(x)
        out = weight + x
        return out

    """
    #在这个特定的例子中，空间注意力机制 (self.spatial_attention(x)) 和通道注意力机制 (self.channel_attention(x))
    #都输出一个与输入特征图 (x) 形状相同的张量。这两个输出张量中的每个元素都是一个注意力权重，它们分别表示空间和通道维度上的注意力分布。
    #逐元相乘操作将这两个输出张量相乘，从而将空间和通道注意力权重结合在一起。这样，我们可以在一个操作中同时捕捉空间和通道维度上的上下文信息。
    #最后，将注意力权重逐元素地应用于输入特征图 x。这意味着每个特征图的值都会根据空间和通道注意力权重进行调整。
    #这样，经过注意力机制调整的特征图能更好地捕捉输入数据中的重要信息。
    """
if __name__ == "__main__":
    input = torch.randn(1, 128, 20, 20)
    models = MultiScaleAttention(128, 128, 1)
    print(models(input).shape)
