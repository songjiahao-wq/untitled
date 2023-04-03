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
class Focus(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Focus, self).__init__()
        self.conv = nn.Conv2d(in_channels * 4, out_channels, kernel_size=1)

    def forward(self, x):
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], dim=1)
        return self.conv(x)
# class Focus(nn.Module):
#     # Focus wh information into c-space
#     def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
#         super().__init__()
#         self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
#
#         # self.contract = Contract(gain=2)
#
#     def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
#         return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
#         # return self.conv(self.contract(x))
class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()
        self.up_sample = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up_sample(x)

class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()
        self.down_sample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.down_sample(x)

class FusionModel(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(FusionModel, self).__init__()
        self.focus = Focus(in_channels, middle_channels)
        self.up_sample = UpSample(out_channels, middle_channels)
        self.concat = nn.Conv2d(middle_channels * 3, middle_channels, kernel_size=1)

    def forward(self, x1, x2, x3):
        x1 = self.focus(x1)
        x1_up = self.up_sample(x3)
        x = torch.cat([x1, x1_up, x2], dim=1)
        return self.concat(x)

in_channels = 64
middle_channels = 128
out_channels = 256
input1 = torch.randn(1, in_channels, 256, 256)
input2 = torch.randn(1, middle_channels, 128, 128)
input3 = torch.randn(1, out_channels, 64, 64)

fusion_model = FusionModel(in_channels, middle_channels, out_channels)
output = fusion_model(input1, input2, input3)
print(output.shape)  # Should be [1, out_channels, 128, 128]
model  = Focus(64, 128)
print(model(input1).shape)