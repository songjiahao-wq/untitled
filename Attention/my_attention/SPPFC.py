import torch
import torch.nn as nn


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """标准卷积层定义"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)


class SPPFC(nn.Module):
    def __init__(self, c1, c2, k=[3, 3, 3], reduction=16, group=True,
                 conv_groups=[4, 8, 16, 32]):
        super(SPPFC, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = conv(c1, c_, 1, 1)
        self.cv2 = conv(c_ * (1 + len(k)), c2 // reduction, 1, 1)

        self.m = nn.ModuleList([
            conv(c_, c_, k[i], padding=k[i] // 2, groups=(conv_groups[i] if group else 1))
            for i in range(len(k))
        ])

        self.fc = nn.Sequential(
            nn.Linear(c2 // reduction, c2 // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(c2 // reduction, c1, bias=False),  # 这里需要输出通道数与输入通道数一致
            nn.Sigmoid()
        )
        self.avg_pool1 = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        b, c, _, _ = x.shape
        x1 = self.cv1(x)
        z = [x1]

        for m in self.m:
            z.append(m(z[-1]))

        out = self.cv2(torch.cat(z, 1))
        out = self.avg_pool1(out).view(b, -1)
        weight = self.fc(out).view(b, c, 1, 1)
        return x * weight


# 测试代码
if __name__ == "__main__":
    model = SPPFC(c1=64, c2=128, k=[3, 5, 7], reduction=16, group=True, conv_groups=[4, 8, 16])
    x = torch.randn(8, 64, 32, 32)  # batch size 8, 64 channels, 32x32 image
    output = model(x)
    print(output.shape)
