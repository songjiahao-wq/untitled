import os
import random
import shutil

def random_select_images_labels(src_image_folder, src_label_folder, dst_image_folder, dst_label_folder, num_samples):
    # 获取图片和标注文件列表
    image_files = [f for f in os.listdir(src_image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
    label_files = [f.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt') for f in image_files]

    # 随机抽取指定数量的样本
    selected_files = random.sample(list(zip(image_files, label_files)), num_samples)

    # 在目标文件夹中创建子文件夹，如果它们不存在
    os.makedirs(dst_image_folder, exist_ok=True)
    os.makedirs(dst_label_folder, exist_ok=True)

    # 复制选定的图片和标注文件到目标文件夹
    for image_file, label_file in selected_files:
        shutil.copy(os.path.join(src_image_folder, image_file), os.path.join(dst_image_folder, image_file))
        shutil.copy(os.path.join(src_label_folder, label_file), os.path.join(dst_label_folder, label_file))

rootpath = r''
# 示例用法
src_image_folder = rootpath+ 'path/to/your/images/folder'
src_label_folder = rootpath+ 'path/to/your/labels/folder'
dst_image_folder = rootpath+ 'path/to/your/destination/images/folder'
dst_label_folder = rootpath+ 'path/to/your/destination/labels/folder'
num_samples = 10000

random_select_images_labels(src_image_folder, src_label_folder, dst_image_folder, dst_label_folder, num_samples)
