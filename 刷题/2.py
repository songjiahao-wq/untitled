import torch.nn as nn
import torch
def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)
class SPPFC(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=[3,5,7] , reduction=16, group=False,conv_groups=[1, 4, 8, 16]):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = conv(c1, c_, 1, 1)
        self.cv2 = conv(c_ * (1 + len(k)), c2, 1, 1)
        if not group:
            self.m = nn.ModuleList(conv(c_, c_, k[i],padding=k[i]//2) for i in range(len(k)))
        else:
            self.m = nn.ModuleList(conv(c_, c_, k[i], padding=k[i] // 2,groups=conv_groups[i]) for i in range(len(k)))
        self.fc = nn.Sequential(
            nn.Linear(c2 , c2 // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(c2 // reduction, c2, bias=False),
            nn.Sigmoid()
        )
        self.avg_pool1 = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        print(self.m)
        b, c, _, _ = x.shape
        x1 = self.cv1(x)
        z = list(torch.unsqueeze(x1, dim=0))

        z.extend(m(z[-1]) for m in self.m)
        out = self.cv2(torch.cat(z, 1))
        out = self.avg_pool1(out).reshape((b, -1))
        weight = self.fc(out).view(b, c, 1, 1)
        return x * weight

if __name__ =="__main__":
    x = torch.randn(1, 64, 20, 20)
    b, c, h, w = x.shape
    net = SPPFC(64, 64)
    y = net(x)
    print(y.size())