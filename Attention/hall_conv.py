import torch
import torch.nn as nn
from Attention import Conv,DWConv
from thop import clever_format, profile

class half_conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1):
        super(half_conv, self).__init__()
        self.c_ = c1 // 2
        self.conv3x3 = DWConv(self.c_, self.c_, 3, 1, 3//2)
        self.conv1x1 = Conv(c1, self.c_, 1, 1)
        self.convout = DWConv(self.c_, self.c_, 3, 2, 3//2)
        self.Conv = DWConv(c1, c2, k, s)
    def forward(self, x):
        y = torch.split(x, [self.c_, self.c_], dim=1)
        identity1 = y[1]
        identity2 = y[0]
        y1 =self.conv3x3(y[0])
        out = torch.cat((y1, identity1), dim=1)
        halfout = self.conv1x1(out)

        return torch.cat((halfout, identity2), 1)
class half_conv_2(nn.Module):
    def __init__(self, c1, c2, k=1, s=1):
        super().__init__()
        self.c_ = c1 // 2
        self.conv3x3 = DWConv(self.c_, self.c_, 3, 1, 3//2)
        self.conv1x1 = Conv(c1, self.c_, 1, 1)
        self.convout = DWConv(c1, self.c_, 3, 1, 3//2)
        self.Conv = DWConv(c1, c2, k, s)
    def forward(self, x):
        y = torch.split(x, [self.c_, self.c_], dim=1)
        identity1 = y[1]
        identity2 = y[0]
        y1 =self.conv3x3(y[0])
        out = torch.cat((y1, identity1), dim=1)
        halfout = self.convout(out)

        return torch.cat((halfout, identity2), 1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dummy_input = torch.randn(1, 256, 40, 40).to(device)
for model_name in [half_conv,half_conv_2]:
    model = model_name(256, 256, 3, 1)
    flops, params = profile(model.to(device), (dummy_input,), verbose=False)
    flops, params = clever_format([flops * 2, params], "%.3f")
    print('Total FLOPS: %s' % (flops))
    print('Total params: %s' % (params))

if __name__ == "__main__":
    input = torch.randn(1, 256, 40, 40)
    model = half_conv(256, 256)
    print(model(input).shape)