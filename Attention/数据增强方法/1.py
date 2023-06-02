import cv2
import os
import numpy as np
import random

# 设置参数
dataset_path = r'D:\yanyi\project_process\datasets\coco128'
output_path = r'D:\yanyi\project_process\datasets\output'
num_per_image = 5  # 每张大图贴几个小图
min_iou = 0  # 最小iou阈值
max_tries = 50  # 最大尝试次数


def main():
    # 读取coco128标签文件
    with open(os.path.join(dataset_path, 'train.txt'), 'r') as f:
        lines = f.readlines()

    # 遍历所有图片
    for line in lines:
        # 读取图片和标签信息
        img_path, _, labels = line.strip().split(' ')
        img = cv2.imread(os.path.join(dataset_path, img_path))
        h, w, _ = img.shape
        labels = labels.split(',')

        # 抠出所有目标并保存到列表中
        objs = []
        for label in labels:
            obj = label.split(',')
            x1, y1, x2, y2, cls_id = map(int, obj)
            obj_img = img[y1:y2, x1:x2]
            objs.append((obj_img, cls_id))

        # 创建输出文件名
        filename, ext = os.path.splitext(img_path)
        output_img_path = os.path.join(output_path, filename + '_aug' + ext)
        output_label_path = os.path.join(output_path, filename + '_aug.txt')

        # 初始化输出图像和标签信息
        output_img = np.zeros((h, w, 3), dtype=np.uint8)
        output_labels = []

        # 随机粘贴小图
        for i in range(num_per_image):
            obj_img, cls_id = random.choice(objs)
            obj_h, obj_w, _ = obj_img.shape

            # 随机选择插入位置
            x = random.randint(0, w - obj_w)
            y = random.randint(0, h - obj_h)

            # 检查是否与大图上的目标有重叠（iou=0）
            overlap = False
            for label in output_labels:
                x1, y1, x2, y2, _ = label
                iou = calculate_iou((x1, y1, x2, y2), (x, y, x + obj_w, y + obj_h))
                if iou > min_iou:
                    overlap = True
                    break

            # 如果有重叠则尝试缩小小图
            tries = 0
            while overlap and tries < max_tries:
                scale = random.uniform(0.5, 1)
                new_obj_img = cv2.resize(obj_img, None, fx=scale, fy=scale)
                new_h, new_w, _ = new_obj_img.shape
                x = random.randint(0, w - new_w)
                y = random.randint(0, h - new_h)
                overlap = False
                for label in output_labels:
                    x1, y1, x2, y2, _ = label
                    iou = calculate_iou((x1, y1, x2, y2), (x, y, x + new_w, y + new_h))
                    if iou > min_iou:
                        overlap = True
                        break
                tries += 1
                obj_img = new_obj_img

            # 如果尝试了多次仍然有重叠则跳过
            if overlap:
                continue

            # 将小图粘贴到大图上并保存标签信息
            output_img[y:y+obj_h, x:x+obj_w] = obj_img
            output_labels.append((x, y, x + obj_w, y + obj_h, cls_id))

        # 保存增强后的图片和标签信息
        cv2.imwrite(output_img_path, output_img)
        with open(output_label_path, 'w') as f:
            for label in output_labels:
                x1, y1, x2, y2, cls_id = label
                f.write(f'{x1},{y1},{x2},{y2},{cls_id}\n')


def calculate_iou(box1, box2):
    """计算两个框的iou"""
    x11, y11, x12, y12 = box1
    x21, y21, x22, y22 = box2
    x_left = max(x11, x21)
    y_top = max(y11, y21)
    x_right = min(x12, x22)
    y_bottom = min(y12, y22)
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (x12 - x11) * (y12 - y11)
    box2_area = (x22 - x21) * (y22 - y21)
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou

if __name__ == 'main':
    main()