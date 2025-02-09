# -*- coding: utf-8 -*-
# @Time    : 2024/12/6 11:18
# @Author  : sjh
# @Site    : 
# @File    : 保存和读取rgb和depth视频.py
# @Comment :
import traceback
import cv2
import numpy as np
import os


class VideoWriter_DEPTH:
    def __init__(self, save_video_path):
        # 初始化视频写入器，使用 FFV1 编解码器（无损）
        fourcc = cv2.VideoWriter_fourcc(*'FFV1')  # 使用 FFV1 编解码器
        fps = 30  # 每秒帧数
        frame_size = (640, 480)  # 深度图像的尺寸 (宽度, 高度)
        video_high_filename = os.path.join(save_video_path, "depth_video_high.avi")
        video_low_filename = os.path.join(save_video_path, "depth_video_low.avi")
        self.video_writer_high = cv2.VideoWriter(video_high_filename, fourcc, fps, frame_size, isColor=False)
        self.video_writer_low = cv2.VideoWriter(video_low_filename, fourcc, fps, frame_size, isColor=False)

    def write_one_frame(self, last_frame):
        if self.video_writer_high is not None and last_frame is not None and self.video_writer_low is not None:
            try:
                # 将 uint16 转换为两个 uint8
                high_bits = (last_frame >> 8).astype(np.uint8)  # 高 8 位
                low_bits = (last_frame & 0xFF).astype(np.uint8)  # 低 8 位
                self.video_writer_high.write(high_bits)
                self.video_writer_low.write(low_bits)
            except Exception as e:
                print(traceback.format_exc())
                print('=' * 60)
                print(type(last_frame))
                print(last_frame.shape)
                print('=' * 60)

    def close_writer(self):
        if self.video_writer_high is not None:
            self.video_writer_high.release()
            self.video_writer_low.release()

class VideoWriter_RGB:
    def __init__(self, save_video_path):
        print(save_video_path)
        # 初始化视频写入器，使用 FFV1 编解码器（无损）
        fourcc = cv2.VideoWriter_fourcc(*'FFV1')  # 使用 FFV1 编解码器
        fps = 30  # 每秒帧数
        frame_size = (640, 480)  # 深度图像的尺寸 (宽度, 高度)
        video_high_filename = os.path.join(save_video_path, "color_video.avi")
        self.writer = cv2.VideoWriter(video_high_filename, fourcc, fps, frame_size, isColor=True)

    def write_one_frame(self, last_frame):
        if self.writer is not None and last_frame is not None:
            try:
                self.writer.write(last_frame)
            except Exception as e:
                print(traceback.format_exc())
                print('=' * 60)
                print(type(last_frame))
                print(last_frame.shape)
                print('=' * 60)
    def close_writer(self):
        if self.writer is not None:
            self.writer.release()

def decode_uint16_from_videos(video_high_filename, video_low_filename):
    cap_high = cv2.VideoCapture(video_high_filename)
    cap_low = cv2.VideoCapture(video_low_filename)

    while cap_high.isOpened() and cap_low.isOpened():
        ret_high, frame_high = cap_high.read()
        ret_low, frame_low = cap_low.read()

        if not ret_high or not ret_low:
            break

        # 将高 8 位和低 8 位组合回 uint16
        frame_uint16 = (frame_high.astype(np.uint16) << 8) | frame_low.astype(np.uint16)
        print(frame_uint16.dtype)
        # 使用opencv显示重建的 uint16 图像
        cv2.imshow('Reconstructed Depth Image', frame_uint16.astype(np.float32) / 65535.0)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap_high.release()
    cap_low.release()
    cv2.destroyAllWindows()
