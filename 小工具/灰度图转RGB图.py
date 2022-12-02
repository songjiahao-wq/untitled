import cv2

# path = r"D:\songjiahao\DATA\smokke\train\images\000120.jpg"
#
# pic = cv2.imread(path,1)
# print(pic)
# import cv2
# from PIL import Image
# import matplotlib.pyplot as plt
# import os
# import numpy as np


import glob
from PIL import Image

for filename in glob.glob(r'D:\songjiahao\DATA\smokke\train\images/*.jpg'):
    img=Image.open(filename).convert("RGB")
    img.save(filename)#原地保存
