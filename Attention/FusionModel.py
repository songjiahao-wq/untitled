import torch
import torch.nn as nn

class Focus(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Focus, self).__init__()
        self.conv = nn.Conv2d(in_channels * 4, out_channels, kernel_size=1)

    def forward(self, x):
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], dim=1)
        return self.conv(x)

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
        self.down_sample = DownSample(middle_channels, middle_channels)
        self.up_sample = UpSample(middle_channels, middle_channels)
        self.concat = nn.Conv2d(middle_channels * 3, out_channels, kernel_size=1)

    def forward(self, x1, x2):
        x1 = self.focus(x1)
        x1_down = self.down_sample(x1)
        x1_up = self.up_sample(x1_down)
        x = torch.cat([x1, x1_up, x2], dim=1)
        return self.concat(x)

in_channels = 3
middle_channels = 64
out_channels = 256
input1 = torch.randn(1, in_channels, 256, 256)
input2 = torch.randn(1, middle_channels, 128, 128)

fusion_model = FusionModel(in_channels, middle_channels, out_channels)
output = fusion_model(input1, input2)
print(output.shape)  # Should be [1, out_channels, 128, 128]
