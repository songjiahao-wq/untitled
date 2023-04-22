import torch.nn as nn
import torch
import torch.nn.functional as F
class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class CAConv(nn.Module):
    def __init__(self, inp, oup, kernel_size, stride, reduction=16):
        super(CAConv, self).__init__()
        self.pool_h1 = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w1 = nn.AdaptiveAvgPool2d((1, None))

        self.pool_h2 = nn.AdaptiveAvgPool2d((None, 2))
        self.pool_w2 = nn.AdaptiveAvgPool2d((2, None))

        self.pool_h3 = nn.AdaptiveAvgPool2d((None, 4))
        self.pool_w3 = nn.AdaptiveAvgPool2d((4, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv = nn.Sequential(nn.Conv2d(inp, oup, kernel_size, padding=kernel_size // 2, stride=stride),
                                  nn.BatchNorm2d(oup),
                                  nn.SiLU())
    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h1 = self.pool_h1(x) #n,c,h,1
        x_w1 = self.pool_w1(x).permute(0, 1, 3, 2)  #n,c,h,1

        x_h2 = self.pool_h2(x) #n,c,h,1
        x_w2 = self.pool_w2(x).permute(0, 1, 3, 2)  #n,c,h,1

        x_h3 = self.pool_h3(x) #n,c,h,1
        x_w3 = self.pool_w3(x).permute(0, 1, 3, 2)  #n,c,h,1

        x_h2 = F.interpolate(x_h2, size=(h, 1), mode='nearest')
        x_w2 = F.interpolate(x_w2, size=(w, 1), mode='nearest')

        x_h3 = F.interpolate(x_h3, size=(h, 1), mode='nearest')
        x_w3 = F.interpolate(x_w3, size=(w, 1), mode='nearest')

        x_h = x_h1 + x_h2 + x_h3
        x_w = x_w1 + x_w2 + x_w3

        y = torch.cat([x_h, x_w], dim=2) #n,c,2h,1
        y = self.conv1(y)  #n,c/16,2h,1
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2) #n,c/16,h,1
        x_w = x_w.permute(0, 1, 3, 2) #n,c/16,1,w

        a_h = self.conv_h(x_h).sigmoid()   #n,c,1,w
        a_w = self.conv_w(x_w).sigmoid() #n,c,1,w

        out = identity * a_w * a_h

        return self.conv(out)
if __name__ == '__main__':
    x = torch.randn(1, 256, 80, 80)#用来生成随机数字的tensor 输出一个64*32*48的张量

    b, c, h, w = x.shape
    net = CAConv(c, c, 3, 2)
    y = net(x)
    print(y.shape)
