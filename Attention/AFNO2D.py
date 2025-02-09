# -*- coding: utf-8 -*-
# @Time    : 2024/10/28 10:37
# @Author  : sjh
# @Site    : 
# @File    : AFNO2D.py
# @Comment :
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.

import math
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F

class AFNO2D(nn.Module):
    """
    hidden_size: channel dimension size
    num_blocks: how many blocks to use in the block diagonal weight matrices (higher => less complexity but less parameters)
    sparsity_threshold: lambda for softshrink
    hard_thresholding_fraction: how many frequencies you want to completely mask out (lower => hard_thresholding_fraction^2 less FLOPs)
    """
    def __init__(self, hidden_size, num_blocks=8, sparsity_threshold=0.01, hard_thresholding_fraction=1, hidden_size_factor=1):
        super().__init__()
        assert hidden_size % num_blocks == 0, f"hidden_size {hidden_size} should be divisble by num_blocks {num_blocks}"

        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02

        self.w1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))

    def forward(self, x, spatial_size=None):
        bias = x

        dtype = x.dtype
        x = x.float()
        B, N, C = x.shape

        if spatial_size == None:
            H = W = int(math.sqrt(N))
        else:
            H, W = spatial_size

        x = x.reshape(B, H, W, C)
        x = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")
        x = x.reshape(B, x.shape[1], x.shape[2], self.num_blocks, self.block_size)

        o1_real = torch.zeros([B, x.shape[1], x.shape[2], self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o1_imag = torch.zeros([B, x.shape[1], x.shape[2], self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)

        total_modes = N // 2 + 1
        kept_modes = int(total_modes * self.hard_thresholding_fraction)

        o1_real[:, :, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, :, :kept_modes].real, self.w1[0]) - \
            torch.einsum('...bi,bio->...bo', x[:, :, :kept_modes].imag, self.w1[1]) + \
            self.b1[0]
        )

        o1_imag[:, :, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, :, :kept_modes].imag, self.w1[0]) + \
            torch.einsum('...bi,bio->...bo', x[:, :, :kept_modes].real, self.w1[1]) + \
            self.b1[1]
        )

        o2_real[:, :, :kept_modes] = (
            torch.einsum('...bi,bio->...bo', o1_real[:, :, :kept_modes], self.w2[0]) - \
            torch.einsum('...bi,bio->...bo', o1_imag[:, :, :kept_modes], self.w2[1]) + \
            self.b2[0]
        )

        o2_imag[:, :, :kept_modes] = (
            torch.einsum('...bi,bio->...bo', o1_imag[:, :, :kept_modes], self.w2[0]) + \
            torch.einsum('...bi,bio->...bo', o1_real[:, :, :kept_modes], self.w2[1]) + \
            self.b2[1]
        )

        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)
        x = x.reshape(B, x.shape[1], x.shape[2], C)
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm="ortho")
        x = x.reshape(B, N, C)
        x = x.type(dtype)
        return x + bias
# Initialize the AFNO2D model with test parameters
hidden_size = 16  # Example hidden size
num_blocks = 4    # Example number of blocks
sparsity_threshold = 0.01
hard_thresholding_fraction = 0.5
hidden_size_factor = 1

# Instantiate the AFNO2D model
model = AFNO2D(
    hidden_size=hidden_size,
    num_blocks=num_blocks,
    sparsity_threshold=sparsity_threshold,
    hard_thresholding_fraction=hard_thresholding_fraction,
    hidden_size_factor=hidden_size_factor
)

# Create a sample input tensor
# Shape: (Batch, Num points, Channels), here we assume (1, 64, 16)
# Batches = 1, Num points = 64 (e.g., 8x8 grid), Channels = hidden_size
sample_input = torch.randn(1, 64, hidden_size)

# Forward pass through the model
output = model(sample_input)

# Print the output shape and content
print("Output Shape:", output.shape)
print("Output:", output)


import torch

# 假设我们有一个尺寸为 (1, 3, 32, 32) 的图像输入 (Batch, Channels, Height, Width)
batch_size, channels, height, width = 1, 3, 32, 32
image_input = torch.randn(batch_size, channels, height, width)

# 初始化 AFNO2D 模型参数
hidden_size = 3  # 必须与 channels 的数量一致
num_blocks = 1
sparsity_threshold = 0.01
hard_thresholding_fraction = 0.5
hidden_size_factor = 1

# 实例化 AFNO2D 模型
model = AFNO2D(
    hidden_size=hidden_size,
    num_blocks=num_blocks,
    sparsity_threshold=sparsity_threshold,
    hard_thresholding_fraction=hard_thresholding_fraction,
    hidden_size_factor=hidden_size_factor
)

# 调整输入形状: (Batch, Channels, Height, Width) -> (Batch, Height*Width, Channels)
image_input = image_input.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)

# 前向传播
output = model(image_input, spatial_size=(height, width))

# 调整输出形状回到图像格式: (Batch, Height*Width, Channels) -> (Batch, Channels, Height, Width)
output = output.view(batch_size, height, width, channels).permute(0, 3, 1, 2)

print("Output Shape:", output.shape)
print("Output:", output)
