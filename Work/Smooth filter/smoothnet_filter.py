# -*- coding: utf-8 -*-
# @Time    : 2024/10/28 18:11
# @Author  : sjh
# @Site    : 
# @File    : smoothnet_filter.py
# @Comment :
# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import numpy as np
import torch
from torch import Tensor, nn

class SmoothNetResBlock(nn.Module):
    """Residual block module used in SmoothNet.

    Args:
        in_channels (int): Input channel number.
        hidden_channels (int): The hidden feature channel number.
        dropout (float): Dropout probability. Default: 0.5
    """
    def __init__(self, in_channels, hidden_channels, dropout=0.5):
        super().__init__()
        self.linear1 = nn.Linear(in_channels, hidden_channels)
        self.linear2 = nn.Linear(hidden_channels, in_channels)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(p=dropout, inplace=True)

    def forward(self, x):
        identity = x
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.lrelu(x)
        x = self.linear2(x)
        x = self.dropout(x)
        x = self.lrelu(x)
        out = x + identity
        return out


class SmoothNet(nn.Module):
    """SmoothNet is a plug-and-play temporal-only network to refine human
    poses. It works for 2d/3d/6d pose smoothing.

    Args:
        window_size (int): The size of the input window.
        output_size (int): The size of the output window.
        hidden_size (int): The hidden feature dimension in the encoder,
            the decoder and between residual blocks. Default: 512
        res_hidden_size (int): The hidden feature dimension inside the
            residual blocks. Default: 256
        num_blocks (int): The number of residual blocks. Default: 3
        dropout (float): Dropout probability. Default: 0.5
    """
    def __init__(self,
                 window_size: int,
                 output_size: int,
                 hidden_size: int = 512,
                 res_hidden_size: int = 256,
                 num_blocks: int = 3,
                 dropout: float = 0.5):
        super().__init__()
        self.window_size = window_size
        self.output_size = output_size

        # Build encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(window_size, hidden_size),
            nn.LeakyReLU(0.1, inplace=True))

        # Build residual blocks
        res_blocks = [SmoothNetResBlock(hidden_size, res_hidden_size, dropout) for _ in range(num_blocks)]
        self.res_blocks = nn.Sequential(*res_blocks)

        # Build decoder layers
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, x: Tensor) -> Tensor:
        N, C, T = x.shape
        num_windows = T - self.window_size + 1

        # Unfold x to obtain input sliding windows [N, C, num_windows, window_size]
        x = x.unfold(2, self.window_size, 1)

        # Forward layers
        x = self.encoder(x)
        x = self.res_blocks(x)
        x = self.decoder(x)  # [N, C, num_windows, output_size]

        # Accumulate output ensembles
        out = x.new_zeros(N, C, T)
        count = x.new_zeros(T)

        for t in range(num_windows):
            out[..., t:t + self.output_size] += x[:, :, t]
            count[t:t + self.output_size] += 1.0

        return out.div(count)


class SmoothNetFilter:
    """Apply SmoothNet filter for pose sequence smoothing.

    Args:
        window_size (int): The size of the filter window.
        output_size (int): The output window size of SmoothNet model.
        hidden_size (int): SmoothNet argument. Default: 512
        res_hidden_size (int): SmoothNet argument. Default: 256
        num_blocks (int): SmoothNet argument. Default: 3
        device (str): Device for model inference. Default: 'cpu'
        root_index (int, optional): If not None, calculates relative keypoint coordinates
                                    by centering around the root point. Default: None
    """
    def __init__(self,
                 window_size: int,
                 output_size: int,
                 hidden_size: int = 512,
                 res_hidden_size: int = 256,
                 num_blocks: int = 3,
                 device: str = 'cpu',
                 root_index: Optional[int] = None):
        self.device = device
        self.root_index = root_index
        self.smoothnet = SmoothNet(window_size, output_size, hidden_size,
                                   res_hidden_size, num_blocks)
        self.smoothnet.to(device)
        self.smoothnet.eval()

        for p in self.smoothnet.parameters():
            p.requires_grad_(False)

    def __call__(self, x: np.ndarray):
        assert x.ndim == 3, ('Input should be an array with shape [T, K, C], '
                             f'but got invalid shape {x.shape}')

        root_index = self.root_index
        if root_index is not None:
            x_root = x[:, root_index:root_index + 1]
            x = np.delete(x, root_index, axis=1)
            x = x - x_root

        T, K, C = x.shape

        if T < self.smoothnet.window_size:
            # Skip smoothing if the input length is less than the window size
            smoothed = x
        else:
            dtype = x.dtype

            # Convert to tensor and forward the model
            with torch.no_grad():
                x = torch.tensor(x, dtype=torch.float32, device=self.device)
                x = x.view(1, T, K * C).permute(0, 2, 1)  # to [1, KC, T]
                smoothed = self.smoothnet(x)  # [1, KC, T]

            # Convert model output back to input shape and format
            smoothed = smoothed.permute(0, 2, 1).view(T, K, C)  # to [T, K, C]
            smoothed = smoothed.cpu().numpy().astype(dtype)

        if root_index is not None:
            smoothed += x_root
            smoothed = np.concatenate(
                (smoothed[:, :root_index], x_root, smoothed[:, root_index:]),
                axis=1)

        return smoothed

import numpy as np
import torch

# 假设输入数据 `x` 形状为 [T, K, C]
# T 是时间步长，K 是节点数量，C 是每个节点的特征数（例如坐标维度）
T, K, C = 100, 17, 3  # 例如 T=100, K=17, C=3 代表17个关键点，每个关键点3维坐标
np.random.seed(42)
x = np.random.randn(T, K, C)  # 生成随机数据作为示例

# 创建 SmoothNetFilter 实例
window_size = 64
output_size = 32
hidden_size = 512
res_hidden_size = 256
num_blocks = 3
device = 'cpu'  # 如果有GPU，可以将设备改为 'cuda'
smoothnet_filter = SmoothNetFilter(
    window_size=window_size,
    output_size=output_size,
    hidden_size=hidden_size,
    res_hidden_size=res_hidden_size,
    num_blocks=num_blocks,
    device=device
)

# 对数据进行平滑处理
smoothed_data = smoothnet_filter(x)

# 输出结果
print("Original data shape:", x.shape, x)
print("Smoothed data shape:", smoothed_data.shape, smoothed_data)
