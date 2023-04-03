import torch.nn as nn
import torch

class SeLayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SeLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = y.view(b,c)
        # y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = torch.clamp(y, 0, 1)
        return x * y
# class SELayer(nn.Module):
#     def __init__(self, channel, reduction=4):
#         super(SELayer, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(channel, channel // reduction),
#             nn.ReLU(inplace=True),
#             nn.Linear(channel // reduction, channel),
#             h_sigmoid()
#         )
#
#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y = self.avg_pool(x)
#         y = y.view(b, c)
#         y = self.fc(y).view(b, c, 1, 1)
#         return x * y
# class SqueezeExcite(nn.Module):
#     def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
#                  act_layer=nn.ReLU, gate_fn=hard_sigmoid, divisor=4, **_):
#         super(SqueezeExcite, self).__init__()
#         self.gate_fn = gate_fn
#         reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
#         self.act1 = act_layer(inplace=True)
#         self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)
#
#     def forward(self, x):
#         x_se = self.avg_pool(x)
#         x_se = self.conv_reduce(x_se)
#         x_se = self.act1(x_se)
#         x_se = self.conv_expand(x_se)
#         x = x * self.gate_fn(x_se)
#         return x
if __name__ =='__main__':
    x = torch.randn(1, 64, 32, 32)  # 用来生成随机数字的tensor 输出一个64*32*48的张量
    b, c, h, w = x.shape
    net1 = SeLayer(c, c)
    # net2 = SELayer(c, c)
    y1 = net1(x)
    print(y1.shape)
