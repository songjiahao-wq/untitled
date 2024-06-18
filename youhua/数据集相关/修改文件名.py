import os
from pathlib import Path

# 原始目录路径
ori_path = r'D:\BaiduSyncdisk\work\YOLO\YOLOv8_trt\ultralytics\runs\detect\predict\train2017'

# 获取目录下的所有文件和文件夹列表
dir_list = os.listdir(ori_path)

# 遍历列表中的每个项
for i in dir_list:
    # 构造文件的完整路径
    old_name = os.path.join(ori_path, i)

    # 检查文件扩展名是否为'.png'
    # 构造新的文件名，将扩展名改为'.jpg'
    if not i.lower().endswith('.jpg'):
        new_name = os.path.join(ori_path, Path(i).stem + '.jpg')

        # 重命名文件
        os.rename(old_name, new_name)

        # 打印重命名前后的文件名
        print(f"{old_name} ======> {new_name}")