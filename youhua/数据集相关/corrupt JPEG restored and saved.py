import os
import cv2
# 解决问题： YOLO系列模型训练日志【WARNING】 corrupt JPEG restored and saved/ignoring corrupt image/label:
dataDir = "original/"
saveDir = "train/"
if not os.path.exists(saveDir):
    os.makedirs(saveDir)

for one_pic in os.listdir(dataDir):
    one_path = dataDir + one_pic
    one_img = cv2.imread(one_path)
    new_path = saveDir + one_pic
    cv2.imwrite(new_path, one_img)