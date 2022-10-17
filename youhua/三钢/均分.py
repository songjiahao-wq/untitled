"""把文件下的内容均分为多分"""
import os
import shutil
import math
from pathlib import Path
ori_impath = 'D:/yanyi/shixi/puyang/zifu/images/'

ori_imlist = os.listdir(ori_impath)
num_img = len(ori_imlist) #总数量

split = 7  #分成6份

split_num = math.floor(num_img / split) #均分的每份数量

split_list = [] #已经分完的name
print(split_num)
for i in range(split):

    for name in ori_imlist:
        if name in split_list:
            continue
        split_list.append(name)
        split_path = os.path.join(Path(ori_impath).parent,str(i))
        if not os.path.exists(split_path):
            os.makedirs(split_path)
        shutil.copy(os.path.join(ori_impath, name), split_path)
        if len(split_list) % split_num ==0:
            break
        # print(os.path.join(ori_impath,name),split_path)


print(len(split_list))
for name in ori_imlist:

    if name not in split_list:
        shutil.copy(os.path.join(ori_impath,name),split_path)
        print(len(split_list),name)
        split_list.append(name)