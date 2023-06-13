import torch
import torch.nn as nn
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))
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
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

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
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return self.conv(out)
class EMA_ATT(nn.Module):
    def __init__(self, c1, c2, k=1, s=1,g=2):
        super(EMA_ATT, self).__init__()
        G_mid = c1 // g
        self.G_conv = Conv(c1, G_mid, 1, 1)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1 = nn.Conv2d(c1, G_mid, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(G_mid)
        self.act = h_swish()
        self.sigmoid = nn.Sigmoid()
        self.Conv_3 = Conv(G_mid, G_mid, 3, 1)
        self.avg_pool1 = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(dim=1)
        self.GroupNorm = nn.BatchNorm2d(G_mid)
        self.conv = Conv(G_mid, c2, k, s)
    def forward(self,x):
        G_identity = self.G_conv(x)

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = x_h.sigmoid()
        a_w = x_w.sigmoid()

        re_weight = G_identity * a_w * a_h
        # Cross-spatial learning
        k2_out = self.GroupNorm(re_weight) * \
                 self.softmax(self.avg_pool1(self.Conv_3(G_identity)))
        # identity learning
        p = self.softmax(self.avg_pool1(self.GroupNorm(G_identity)))
        k1_out = self.Conv_3(G_identity) * \
                 self.softmax(self.avg_pool1(self.GroupNorm(G_identity)))

        ATT_weight = (k1_out + k2_out).sigmoid()

        return  self.conv(ATT_weight * G_identity)

if __name__ == '__main__':
    input = torch.randn(1, 512, 20, 20)
    model = EMA_ATT(512,512)
    print(model(input).shape)