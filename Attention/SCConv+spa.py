import torch.nn as nn
import torch
import torch.nn.functional as F
class SPA(nn.Module):
    #多尺度通道注意力
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool1 = nn.AdaptiveAvgPool2d(1)
        self.avg_pool2 = nn.AdaptiveAvgPool2d(2)
        self.avg_pool4 = nn.AdaptiveAvgPool2d(4)

        self.fc = nn.Sequential(
            nn.Linear(channel * 21, channel // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        # 设置可学习权值
        self.w1 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w1.data.fill_(0.5)
    def forward(self, x):
        b, c, _, _ = x.shape
        y1 = self.avg_pool1(x).reshape((b, -1))
        y2 = self.avg_pool2(x).reshape((b, -1))
        y3 = self.avg_pool4(x).reshape((b, -1))
        y = torch.cat((y1, y2, y3), 1)
        y = self.fc(y).reshape((b, c, 1, 1))
        # y = self.fc(y).reshape((b, c, 1, 1)) *self.w1 #添加自适应学习权值，对通道信息增加自适应特征学习
        return y
class SCConv(nn.Module):
    def __init__(self, inplanes, planes, stride, padding, dilation, groups, pooling_r, norm_layer):
        super(SCConv, self).__init__()
        self.spa = SPA(inplanes)  # Add SPA layer
        self.k2 = nn.Sequential(
                    nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r),
                    nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,
                                padding=padding, dilation=dilation,
                                groups=groups, bias=False),
                    norm_layer(planes),
                    )
        self.k3 = nn.Sequential(
                    nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,
                                padding=padding, dilation=dilation,
                                groups=groups, bias=False),
                    norm_layer(planes),
                    )
        self.k4 = nn.Sequential(
                    nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                                padding=padding, dilation=dilation,
                                groups=groups, bias=False),
                    norm_layer(planes),
                    )

    def forward(self, x):
        identity = x

        out = self.spa(torch.add(identity, F.interpolate(self.k2(x), identity.size()[2:]))) # SPA(identity + k2)
        out = torch.mul(self.k3(x), out) # k3 * SPA(identity + k2)
        out = self.k4(out) # k4

        return out
