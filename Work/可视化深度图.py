# -*- coding: utf-8 -*-
# @Time    : 2024/12/9 18:07
# @Author  : sjh
# @Site    : 
# @File    : 可视化深度图.py
# @Comment :
# -*- coding: utf-8 -*-
# @Time    : 2024/4/17 13:29
# @Author  : sjh
# @Site    :
# @File    : 可视化深度图.py
# @Comment :
import cv2
import numpy as np

# 指定深度图像的路径
path = r"D:\BaiduSyncdisk\work\yolov5_tensorRT\yolov5-master-mutiltask\resize\depth\frame_0007.png"

# 读取图像，确保以灰度模式读取，这通常意味着单通道16-bit图像
depth_image = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # 使用IMREAD_UNCHANGED以确保保留原始位深

if depth_image is None:
    print("Failed to load depth image.")
else:
    # 将16-bit深度图像转换为float，以便进行缩放
    depth_image_float = depth_image.astype(np.float32)

    # # 获取深度图像的最大深度和最小深度值
    min_val, max_val = depth_image_float.min(), depth_image_float.max()
    # print("Minimum and Maximum Depth Values:", min_val, max_val)
    #
    # # 归一化深度图像到0-1范围
    depth_normalized = (depth_image_float - min_val) / (max_val - min_val)
    #
    # # 将归一化后的深度图像缩放到0-255并转换为uint8
    depth_scaled = (255 * depth_normalized).astype(np.uint8)

    # # 使用颜色映射显示深度图
    depth_scaled = cv2.applyColorMap(depth_scaled, cv2.COLORMAP_JET)
    depth_scaled = cv2.cvtColor(depth_image_float,cv2.COLOR_GRAY2BGR)
    # 显示图像
    cv2.imshow('Depth Image', depth_scaled)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
