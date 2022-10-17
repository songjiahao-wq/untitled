import os
import xml.etree.ElementTree as ET

dirpath = r'D:\songjiahao\DATA\AugINRIAPerson\labels\train/'  # 原来存放xml文件的目录
newdir = r'D:\songjiahao\DATA\AugINRIAPerson\labels2\train/'  # 修改label后形成的txt目录

if not os.path.exists(newdir):
    os.makedirs(newdir)
for fp in os.listdir(dirpath):
    fp_path = dirpath+fp

    with open(fp_path) as f:
        txt = f.readline()
        print(txt)