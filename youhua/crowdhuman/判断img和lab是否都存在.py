#第一天学习
#判断同一个照片和labels是否都存在
import os


img_path = r'D:\songjiahao\DATA\crowhuman\images\test/'
labels_path = r'D:\songjiahao\DATA\crowhuman\labels\test/'

img_dir = os.listdir(img_path)
for i in img_dir:
    lable_name = i.replace('.jpg','.txt')
    lable_dir = os.listdir(labels_path)
    if lable_name not in lable_dir:
        print(lable_name)