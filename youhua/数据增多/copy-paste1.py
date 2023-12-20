# -*- coding: utf-8 -*-
# @Time    : 2023/12/20 14:58
# @Author  : sjh
# @Site    : 
# @File    : 11.py
# @Comment :
"""
为了实现所述的CopyPaste数据增强流程，我们将开发一个脚本，该脚本能够自动从obj文件夹中读取标签和对应的图像，然后将标签指定的对象区域复制并粘贴到background文件夹中的图像上。
同时，脚本将生成相应的新标签文件。在此过程中，每个背景图像将随机选取最多三个对象进行增强。这些对象可以根据需要进行尺寸调整，以确保它们适合背景图像并且不与背景中已有的对象重叠。以下是详细的实现步骤：
1. 读取标签和图像数据：
    从 obj/labels 读取每个对象的标签信息。从 obj/images 读取对应的对象图像。从 background/images 读取背景图像。
2. 随机选择标签和对象：
    对于每个背景图像，随机选择三个或更少的对象标签。根据选定的标签，找到对应的对象图像。
3. 处理对象图像：
    可以对对象图像进行缩放，以适应背景图像的尺寸。确保缩放后的对象不会与背景中已有的目标重合。
4. 将对象粘贴到背景上：
    将处理过的对象图像粘贴到背景图像上。更新标签信息以反映新的对象位置。
5. 保存增强后的图像和标签：
6. 将增强后的图像保存到一个新的文件夹中。
    将更新后的标签信息保存到对应的标签文件中。
"""
import os
import random
import cv2

def parse_label_file(label_file):
    labels = []
    with open(label_file, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) >= 5:
                labels.append([int(parts[0])] + [float(part) for part in parts[1:]])
    return labels

def resize_object(obj_img, scale_factor):
    height, width = obj_img.shape[:2]
    new_height, new_width = int(height * scale_factor), int(width * scale_factor)
    return cv2.resize(obj_img, (new_width, new_height)), scale_factor

def paste_object(bg_img, obj_img, position):
    y, x = position
    h, w = obj_img.shape[:2]
    bg_img[y:y+h, x:x+w] = obj_img
    return bg_img

def update_label(labels, obj_label, position, scale_factor, img_shape):
    x_center, y_center, width, height = obj_label[1:]
    x_center = (x_center * img_shape[1] + position[1]) / img_shape[1]
    y_center = (y_center * img_shape[0] + position[0]) / img_shape[0]
    width *= scale_factor
    height *= scale_factor
    labels.append([obj_label[0], x_center, y_center, width, height])
    return labels

def save_label(labels, label_path):
    with open(label_path, 'w') as file:
        for label in labels:
            file.write(' '.join(map(str, label)) + '\n')
def extract_object(img, bbox):
    x_center, y_center, width, height = bbox
    x = int((x_center - width / 2) * img.shape[1])
    y = int((y_center - height / 2) * img.shape[0])
    w = int(width * img.shape[1])
    h = int(height * img.shape[0])
    return img[y:y+h, x:x+w]
def copypaste_data_augmentation(background_dir, obj_dir, output_dir, max_objects=1):
    bg_images_dir = os.path.join(background_dir, 'images')
    bg_labels_dir = os.path.join(background_dir, 'labels')
    obj_images_dir = os.path.join(obj_dir, 'images')
    obj_labels_dir = os.path.join(obj_dir, 'labels')

    for bg_img_name in os.listdir(bg_images_dir):
        bg_img_path = os.path.join(bg_images_dir, bg_img_name)
        bg_label_path = os.path.join(bg_labels_dir, bg_img_name.replace('.jpg', '.txt'))
        bg_img = cv2.imread(bg_img_path)
        bg_labels = parse_label_file(bg_label_path)

        obj_names = random.sample(os.listdir(obj_images_dir), max_objects)
        for obj_name in obj_names:
            obj_img_path = os.path.join(obj_images_dir, obj_name)
            obj_label_path = os.path.join(obj_labels_dir, obj_name.replace('.jpg', '.txt'))
            obj_img = cv2.imread(obj_img_path)
            obj_labels = parse_label_file(obj_label_path)
            for obj_label in obj_labels:
                extracted_obj = extract_object(obj_img, obj_label[1:])
                if obj_labels:

                    obj_label = obj_labels[0]
                    obj_img, scale_factor = resize_object(extracted_obj, random.uniform(0.5, 0.8))
                    if obj_img.shape[0] > bg_img.shape[0] or obj_img.shape[1] > bg_img.shape[1]:
                        break
                    max_y = max(bg_img.shape[0] - extracted_obj.shape[0], 0)
                    max_x = max(bg_img.shape[1] - extracted_obj.shape[1], 0)
                    position = (random.randint(0, max_y), random.randint(0, max_x))
                    bg_img = paste_object(bg_img, obj_img, position)
                    bg_labels = update_label(bg_labels, obj_label, position, scale_factor, bg_img.shape)

        output_img_path = os.path.join(output_dir, 'images', bg_img_name)
        output_label_path = os.path.join(output_dir, 'labels', bg_img_name.replace('.jpg', '.txt'))
        cv2.imwrite(output_img_path, bg_img)
        save_label(bg_labels, output_label_path)

copypaste_data_augmentation(r'E:\dddd\vocdevkit\background', r'E:\dddd\vocdevkit\obj', r'E:\dddd\vocdevkit\output')

#args: max_objects调增强个数
# random.uniform(0.5, 0.8))，调缩小的高宽倍数