import torch
import torch.nn as nn
import torch.nn.functional as F
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
    def __init__(self, inp, oup, kernel_size, stride, reduction=32):
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
                                  nn.ReLU())

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
class Conv_L(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.LayerNorm((c2, s, s))  # 修改此处，其中s表示height和width
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class RFCAConv(nn.Module):
    def __init__(self, c1, c2, kernel_size, stride):
        super(RFCAConv, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.group_conv1 = Conv_L(c1, 9 *c1, k=1, g=c1)
        self.group_conv2 = Conv_L(c1, 9 *c1, k=3, g=c1)
        self.group_conv3 = Conv_L(c1, 9 *c1, k=5, g=c1)

        self.softmax = nn.Softmax(dim=1)

        self.group_conv = Conv(c1, 9 * c1, k=3, g=c1)
        self.convDown = Conv(c1, c1, k=3, s=3)
        self.CA = CAConv(c1, c2, kernel_size, stride)
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)

        group1 = self.softmax(self.group_conv1(y))
        group2 = self.softmax(self.group_conv2(y))
        group3 = self.softmax(self.group_conv3(y))
        # g1 =  torch.cat([group1, group2, group3], dim=1)

        g2 = self.group_conv(x)

        out1 = g2 * group1
        out2 = g2 * group2
        out3 = g2 * group3

        out = sum([out1, out2, out3])
        # 获取输入特征图的形状
        batch_size, channels, height, width = out.shape

        # 计算输出特征图的通道数
        output_channels = channels // 9

        # 重塑并转置特征图以将通道数分成3x3个子通道并扩展高度和宽度
        out = out.view(batch_size, output_channels, 3, 3, height, width).permute(0, 1, 4, 2, 5,3).\
                                                reshape(batch_size, output_channels, 3 * height, 3 * width)
        out = self.convDown(out)
        out = self.CA(out)
        return out


# Example usage:

# input_tensor = torch.randn(1, 32, 40, 40)  # Batch x Channels x Height x Width
# channel_attention_module = RFCAConv(c1=32 ,c2=64, kernel_size=3,stride=2).eval()
# output_tensor = channel_attention_module(input_tensor)
# print(output_tensor.shape)

input = torch.randn(1,288,40,40)
batch_size, channels, height, width = input.shape

        # 计算输出特征图的通道数
output_channels = channels // 9
#(1,32,120,120)
out = input.view(1,32,120,120)
out2 = out.view(batch_size, output_channels, 3, 3, height, width).permute(0, 1, 4, 2, 5,3).\
                                                reshape(batch_size, output_channels, 3 * height, 3 * width)
out3 = out.view(batch_size, output_channels, 3, 3, height, width).permute(0, 1, 4, 2, 5,3).\
                                                reshape(batch_size, output_channels, 3 * height, 3 * width)
print(out.shape)
print(out2.shape)
are_tensors_equal = torch.allclose(out, out2)  # True
print(are_tensors_equal)
are_tensors_equal = torch.allclose(out3, out2)  # True
print(are_tensors_equal)