"""

"""
import torch
from torch.nn import functional
import torch.nn as nn
from torch.nn.parameter import Parameter

# class h_swish(nn.Module):
#     def __init__(self, inplace=True):
#         super(h_swish, self).__init__()
#         self.sigmoid = h_sigmoid(inplace=inplace)
#
#     def forward(self, x):
#         return x * self.sigmoid(x)
# class h_sigmoid(nn.Module):
#     def __init__(self, inplace=True):
#         super(h_sigmoid, self).__init__()
#         self.relu = nn.ReLU6(inplace=inplace)
#
#     def forward(self, x):
#         return self.relu(x + 3) / 6
# class CAAttention(nn.Module):
#     # 坐标注意力
#     def __init__(self, inp, oup, reduction=32):
#         super(CAAttention, self).__init__()
#
#         mip = max(8, inp // reduction)
#
#         self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
#         self.bn1 = nn.BatchNorm2d(mip)
#         self.act = h_swish()
#
#         self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
#         self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
#
#     def forward(self, x):
#         identity = x
#
#         n, c, h, w = x.size()
#         pool_h = nn.AdaptiveAvgPool2d((h, 1))
#         pool_w = nn.AdaptiveAvgPool2d((1, w))
#         x_h = pool_h(x)
#         x_w = pool_w(x).permute(0, 1, 3, 2)
#
#         y = torch.cat([x_h, x_w], dim=2)
#         y = self.conv1(y)
#         y = self.bn1(y)
#         y = self.act(y)
#
#         x_h, x_w = torch.split(y, [h, w], dim=2)
#         x_w = x_w.permute(0, 1, 3, 2)
#
#         a_h = self.conv_h(x_h).sigmoid()
#         a_w = self.conv_w(x_w).sigmoid()
#
#         out = identity * a_w * a_h
#
#         return out
# class RSBU_CW(torch.nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, down_sample=False):
#         super().__init__()
#         self.down_sample = down_sample
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         stride = 1
#         if down_sample:
#             stride = 2
#         self.BRC = nn.Sequential(
#             nn.BatchNorm1d(in_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
#                       padding=1),
#             nn.BatchNorm1d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1,
#                       padding=1)
#         )
#         self.global_average_pool = nn.AdaptiveAvgPool2d(1)
#         self.FC = nn.Sequential(
#             nn.Linear(in_features=out_channels, out_features=out_channels//16),
#             nn.BatchNorm1d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Linear(in_features=out_channels//16, out_features=out_channels),
#             nn.Sigmoid()
#         )
#         self.flatten = nn.Flatten()
#         self.average_pool = nn.AvgPool2d(kernel_size=1, stride=2)
#
#
#     def forward(self, input):
#         b,c,_,_ = input.shape
#         x = self.BRC(input)
#         x_abs = torch.abs(x)
#         gap = self.global_average_pool(x_abs).reshape((b, -1))
#         # gap = self.flatten(gap)
#         alpha = self.FC(gap)
#         threshold = torch.mul(gap, alpha)
#         threshold = torch.unsqueeze(threshold, 2)
#         # 软阈值化
#         sub = x_abs - threshold
#         zeros = sub - sub
#         n_sub = torch.max(sub, zeros)
#         x = torch.mul(torch.sign(x), n_sub)
#         if self.down_sample:  # 如果是下采样，则对输入进行平均池化下采样
#             input = self.average_pool(input)
#         if self.in_channels != self.out_channels:  # 如果输入的通道和输出的通道不一致，则进行padding,直接通过复制拼接矩阵进行padding,原代码是通过填充0
#             zero_padding=torch.zeros(input.shape).cuda()
#             input = torch.cat((input, zero_padding), dim=1)
#
#         result = x + input
#         return result
class RSBU_CW(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, down_sample=False):
        super().__init__()
        self.down_sample = down_sample
        self.in_channels = in_channels
        self.out_channels = out_channels
        stride = 1
        if down_sample:
            stride = 2
        self.BRC = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1,
                      padding=1)
        )
        self.global_average_pool = nn.AdaptiveAvgPool1d(1)
        self.FC = nn.Sequential(
            nn.Linear(in_features=out_channels, out_features=out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=out_channels, out_features=out_channels),
            nn.Sigmoid()
        )
        self.flatten = nn.Flatten()
        self.average_pool = nn.AvgPool1d(kernel_size=1, stride=2)

    def forward(self, input):
        x = self.BRC(input)
        x_abs = torch.abs(x)
        gap = self.global_average_pool(x_abs)
        gap = self.flatten(gap)
        alpha = self.FC(gap)
        threshold = torch.mul(gap, alpha)
        threshold = torch.unsqueeze(threshold, 2)
        # 软阈值化
        sub = x_abs - threshold
        zeros = sub - sub
        n_sub = torch.max(sub, zeros)
        x = torch.mul(torch.sign(x), n_sub)
        if self.down_sample:  # 如果是下采样，则对输入进行平均池化下采样
            input = self.average_pool(input)
        if self.in_channels != self.out_channels:  # 如果输入的通道和输出的通道不一致，则进行padding,直接通过复制拼接矩阵进行padding,原代码是通过填充0
            zero_padding=torch.zeros(input.shape).cuda()
            input = torch.cat((input, zero_padding), dim=1)

        result = x + input
        return result


class DRSNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=3, stride=2, padding=1)
        self.bn = nn.BatchNorm1d(16)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.global_average_pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.linear6_8 = nn.Linear(in_features=256, out_features=128)
        self.linear8_4 = nn.Linear(in_features=128, out_features=64)
        self.linear4_2 = nn.Linear(in_features=64, out_features=32)
        self.output_center_pos = nn.Linear(in_features=32, out_features=1)
        self.output_width = nn.Linear(in_features=32, out_features=1)

        self.linear = nn.Linear(in_features=16, out_features=8)
        self.output_class = nn.Linear(in_features=8, out_features=3)

    def forward(self, input):  # 1*256
        x = self.conv1(input)  # 4*128
        x = RSBU_CW(in_channels=4, out_channels=4, kernel_size=3, down_sample=True).cuda()(x)  # 4*64
        x = RSBU_CW(in_channels=4, out_channels=4, kernel_size=3, down_sample=False).cuda()(x)  # 4*64
        x = RSBU_CW(in_channels=4, out_channels=8, kernel_size=3, down_sample=True).cuda()(x)  # 8*32
        x = RSBU_CW(in_channels=8, out_channels=8, kernel_size=3, down_sample=False).cuda()(x)  # 8*32
        x = RSBU_CW(in_channels=8, out_channels=16, kernel_size=3, down_sample=True).cuda()(x)  # 16*16
        x = RSBU_CW(in_channels=16, out_channels=16, kernel_size=3, down_sample=False).cuda()(x)  # 16*16
        x = self.bn(x)
        x = self.relu(x)
        gap = self.global_average_pool(x)  # 16*1
        gap = self.flatten(gap)  # 1*16
        linear1 = self.linear(gap)  # 1*8
        output_class = self.output_class(linear1)  # 1*3
        output_class = self.softmax(output_class)  # 1*3

        return output_class
if __name__ =="__main__":
    x = torch.randn(1, 20, 20)
    x2 = torch.randn(2,128, 20, 20)
    out = torch.add(x, x2)
    model = RSBU_CW(1,4,3)

    print(model(x))
    # print(torch.cat((x,x2),1).shape)
    # print(out.shape)

