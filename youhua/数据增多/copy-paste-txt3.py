# -*- coding: utf-8 -*-
# @Time    : 2023/12/6 14:00
# @Author  : sjh
# @Site    : 
# @File    : 1.py
# @Comment :

import cv2
import os
import random
# 计算两个边界框之间的交并比（Intersection over Union, IoU）
def iou(boxA, boxB):
    """
    Compute the Intersection over Union (IoU) between two bounding boxes.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou
# 在图像上随机粘贴边界框内的对象
def random_paste(img, boxes):
    h, w, _ = img.shape
    new_boxes = boxes.copy()
    extracted_objects = []

    for box in boxes:
        box_id = box[0]  # Extract the original ID
        x_center, y_center, width, height = box[1:]
        x1, y1 = int((x_center - width / 2) * w), int((y_center - height / 2) * h)
        x2, y2 = int((x_center + width / 2) * w), int((y_center + height / 2) * h)
        extracted_objects.append((box_id, (x1, y1, x2, y2), img[y1:y2, x1:x2]))

    for i in range(3):  # paste three times
        if not extracted_objects:
            break
        box_id, _, obj_img = random.choice(extracted_objects)  # Extract the original ID
        oh, ow, _ = obj_img.shape
        tims1 = 0
        while ow > 250 or oh > 250:
            box_id, _, obj_img = random.choice(extracted_objects)  # Extract the original ID
            oh, ow, _ = obj_img.shape
            tims1 += 1
            if tims1 > 15:
                break
        while True:
            if w-ow-1 == 0 or h - oh - 1 == 0 or tims1 > 15:
                break
            x1 = random.randint(0, w - ow - 1)
            y1 = random.randint(0, h - oh - 1)
            x2, y2 = x1 + ow, y1 + oh
            # 随机选择一个新位置，使得对象不超出原图像边界，并确保新位置与 new_boxes 中已有的边界框不重叠（IoU小于0.2）。
            overlap = any([iou((x1, y1, x2, y2), box[1:]) > 0.2 for box in new_boxes])
            if not overlap:
                break
        try:
            img[y1:y2, x1:x2] = obj_img
        except:
            continue
        new_boxes.append([box_id, (x1 + x2) / (2 * w), (y1 + y2) / (2 * h), ow / w, oh / h])  # Use the original ID

    return img, new_boxes


images_dir = r"D:\BaiduNetdiskDownload\KT5\images\train"
labels_dir = r"D:\BaiduNetdiskDownload\KT5\labels\train"
augmented_img_path = r"D:\BaiduNetdiskDownload\KT5\augimages\train"
augmented_txt_path = r"D:\BaiduNetdiskDownload\KT5\auglabels\train"
if not os.path.exists(augmented_img_path):
    os.makedirs(augmented_img_path)
if not os.path.exists(augmented_txt_path):
    os.makedirs(augmented_txt_path)
# Load image and labels
image_files = os.listdir(images_dir)
for image_file in image_files:
    print(image_file)
    # Determine the file extension (either .jpg or .png)
    file_extension = '.jpg' if '.jpg' in image_file else '.png'

    img = cv2.imread(os.path.join(images_dir, image_file))

    # Open corresponding label file based on file extension
    with open(os.path.join(labels_dir, image_file.replace(file_extension, '.txt')), 'r') as f:
        boxes = [list(map(float, line.strip().split())) for line in f]

    augmented_img, augmented_boxes = random_paste(img, boxes)

    # Save augmented image
    cv2.imwrite(os.path.join(augmented_img_path, image_file), augmented_img)

    # Save augmented boxes to text file
    with open(os.path.join(augmented_txt_path, image_file.replace(file_extension, '.txt')), 'w') as f:
        for box in augmented_boxes:
            f.write(" ".join(map(str, box)) + '\n')