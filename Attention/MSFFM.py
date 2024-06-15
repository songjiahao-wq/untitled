# -*- coding: utf-8 -*-
# @Time    : 2024/6/15 10:48
# @Author  : sjh
# @Site    : 
# @File    : MSFFM.py
# @Comment : https://arxiv.org/pdf/2112.13082 多尺度融合模块
import torch
import torch.nn as nn

class MSFFM_AttentionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MSFFM_AttentionModule, self).__init__()
        self.conv_low = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv_high = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, low_level_feat, high_level_feat):
        A = self.conv_low(low_level_feat)  # (B, C, H, W)
        B_feat = self.conv_high(high_level_feat)  # (B, C, H, W)

        # Reshape and transpose
        B, C, H, W = A.size()
        N = H * W
        P = A.view(B, C, N)  # (B, C, N)
        Q = B_feat.view(B, C, N).permute(0, 2, 1)  # (B, N, C)

        # Compute S = Q * P^T
        S = torch.bmm(Q, P)  # (B, N, N)
        S = torch.softmax(S, dim=-1)

        # Compute L = P * S^T
        L = torch.bmm(Q.permute(0,2,1), S.permute(0, 2, 1))  # (B, C, N)
        L = L.view(B, C, H, W)  # (B, C, H, W)

        # Output O = A + L
        O = B_feat + L

        return O

# Example usage
if __name__ == "__main__":
    low_level_feat = torch.randn(1, 512, 40, 40)  # Example low-level feature
    high_level_feat = torch.randn(1, 512, 40, 40)  # Example high-level feature

    attention_module = MSFFM_AttentionModule(in_channels=512, out_channels=512)
    output = attention_module(low_level_feat, high_level_feat)

    print(output.shape)  # Should output torch.Size([1, 64, 32, 32])



