import os
import re


def preprocess_folder(folder_path):
    for filename in os.listdir(folder_path):
        # 获取文件的绝对路径
        file_path = os.path.join(folder_path, filename)

        # 如果是图像文件（以.jpg或.png结尾），则重命名后缀为.jpg
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            new_filename = re.sub(r'\W+', '', filename.split('.')[0]) + '.jpg'
            new_file_path = os.path.join(folder_path, new_filename)
            os.rename(file_path, new_file_path)
            print(f"Renamed image: {filename} -> {new_filename}")

        # 如果是XML文件，检查是否有对应的图像文件，如果没有则删除
        elif filename.lower().endswith('.xml'):
            img_filename = filename.split('.')[0] + '.jpg'
            img_file_path = os.path.join(folder_path, img_filename)
            if not os.path.exists(img_file_path):
                os.remove(file_path)
                print(f"Removed XML without corresponding image: {filename}")

        # 对其他文件（不是图像或XML）去除空格、特殊字符和中文字符

        new_filename = re.sub(r'[^\x00-\x7F]+', '', re.sub(r'\W+', '', filename))
        new_file_path = os.path.join(folder_path, new_filename)
        os.rename(file_path, new_file_path)
        print(f"Renamed other file: {filename} -> {new_filename}")


# 替换为你的图像文件夹和XML文件夹路径
img_folder = '/path/to/your/img_folder'
xml_folder = '/path/to/your/xml_folder'

# 对图像文件夹进行预处理
preprocess_folder(img_folder)

# 对XML文件夹进行预处理
preprocess_folder(xml_folder)
