# 要转换的图片的保存地址，按顺序排好，后面会一张一张按顺序读取。
import glob
import cv2
# import numpy as np

# cv2.__version__
convert_image_path = 'images-out'
frame_rate = 30

# 帧率(fps)，尺寸(size)，size为图片的大小，本文转换的图片大小为1920×1080，
# 即宽为1920，高为1080，要根据自己的情况修改图片大小。
size = (960, 544)
fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V') # mp4

# cv2.VideoWriter_fourcc('I', '4', '2', '0')
videoWriter = cv2.VideoWriter('output.mp4', fourcc,
                              frame_rate, size)
for img in glob.glob(convert_image_path + "/*.jpg"):
    read_img = cv2.imread(img)
    videoWriter.write(read_img)
videoWriter.release()