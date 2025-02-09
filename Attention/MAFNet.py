# -*- coding: utf-8 -*-
# @Time    : 2024/10/15 10:20
# @Author  : sjh
# @Site    : 
# @File    : MAFNet.py
# @Comment :
"""
""
论文《Multiscale Adaptive Fusion Network for Hyperspectral Image Denoising》的主要创新点如下：

1. 多尺度信息聚合框架：提出了一个多尺度自适应融合网络（MAFNet），通过逐步融合多尺度信息，实现了全球上下文信息建模与空间细节保留的同时进行，增强了图像去噪的效果。

2. 联合注意力融合模块：设计了一个联合注意力（coattention）融合模块，能够动态选择来自不同尺度的有用特征，提升了网络的判别学习能力，从而自适应地聚合多尺度信息，并增强了不同尺度间的相关性。

3. 综合实验验证：在多个基准数据集上进行了大量实验，结果表明该模型在合成和真实数据上均优于现有的最先进方法，证明了MAFNet的合理性与有效性。

这些创新点有效解决了超光谱图像去噪中的多尺度特征交互利用和丰富光谱结构保留的挑战​(Multiscale_Adaptive_Fus…)。


"""