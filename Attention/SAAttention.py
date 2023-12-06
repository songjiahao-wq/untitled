# -*- coding: utf-8 -*-
# @Time    : 2023/11/22 16:30
# @Author  : sjh
# @Site    : 
# @File    : SAAttention.py
# @Comment :
import torch
import torch.nn as nn
import torch.nn.functional as F
# https://blog.csdn.net/sinat_17456165/article/details/106913474
# https://arxiv.org/pdf/1909.03402.pdf
class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.conv(x)
        return x

class SqueezeAttentionBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(SqueezeAttentionBlock, self).__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv = conv_block(ch_in, ch_out)
        self.conv_atten = conv_block(ch_in, ch_out)
        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, x):
        # print(x.shape)
        x_res = self.conv(x)
        # print(x_res.shape)
        y = self.avg_pool(x)
        # print(y.shape)
        y = self.conv_atten(y)
        y2 = nn.Sigmoid()(y)
        # print(y.shape)
        y = self.upsample(y2)
        # print(y.shape, x_res.shape)
        return (y * x_res) + y

if __name__ =='__main__':
    input = torch.randn(1,512,40,40)
    model = SqueezeAttentionBlock(512,512)
    print(model(input).shape)