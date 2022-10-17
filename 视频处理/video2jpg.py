"""从视频逐帧读取图片"""
import glob
import cv2
import numpy as np

cv2.__version__
# 读取视频文件
cap = cv2.VideoCapture('./input.mp4')

# 获取视频帧率（30p/60p）
frame_rate = round(cap.get(5))

# 获取视频帧数
frame_num = cap.get(7)

# type(frame_num)
# 由于frame_num是foat型，需要int()转换

# 逐帧获取图片
for i in range(int(frame_num)):
    ret, frame = cap.read()
    cv2.imwrite('images\match_snapshot%d.jpg' % i, frame)

np.shape(frame)
cap.release()
cv2.destroyAllWindows()