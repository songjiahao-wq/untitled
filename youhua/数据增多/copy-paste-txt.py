import os
import random
import cv2
import numpy as np

"""
现在有yolo 的images文件和labels文件，分别存放图像和标签文件，请实现copy-paste数据增强代码，要求：可以将标签为0的行人目标从其他途中提取出来，
随机粘贴到当前图像中，并得到增强后的标签，并且粘贴的目标不能与当前图像中已存在的目标重合。流程为：先提取所有标签为0的行人目标，之后对读取每个图像和标签，
从已提取的标签中随机抽取三个目标，粘贴到当前图像中
"""
def get_boxes_from_label(label_file):
    with open(label_file, 'r') as f:
        lines = f.readlines()
    boxes = [list(map(float, line.strip().split()[1:])) for line in lines if int(line.strip().split()[0]) == 0]
    return boxes


def paste_image(src_img, target_img, box):
    x_center, y_center, w, h = box
    x_center *= src_img.shape[1]
    y_center *= src_img.shape[0]
    w *= src_img.shape[1]
    h *= src_img.shape[0]

    x1, y1, x2, y2 = int(x_center - w/2), int(y_center - h/2), int(x_center + w/2), int(y_center + h/2)

    # Ensure that the coordinates are within the bounds of the src image
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(src_img.shape[1], x2)
    y2 = min(src_img.shape[0], y2)

    person = src_img[y1:y2, x1:x2]

    # Calculate the target paste location
    ty1, ty2 = y1, y1 + (y2 - y1)
    tx1, tx2 = x1, x1 + (x2 - x1)

    # Adjust if it goes out of target image boundaries
    if ty2 > target_img.shape[0]:
        ty2 = target_img.shape[0]
        ty1 = ty2 - (y2 - y1)
    if tx2 > target_img.shape[1]:
        tx2 = target_img.shape[1]
        tx1 = tx2 - (x2 - x1)

    target_img[ty1:ty2, tx1:tx2] = person[:ty2-ty1, :tx2-tx1]




def check_overlap(box1, box2, threshold=0.2):
    x_center1, y_center1, w1, h1 = box1
    x_center2, y_center2, w2, h2 = box2
    x1, y1, x2, y2 = x_center1 - w1 / 2, y_center1 - h1 / 2, x_center1 + w1 / 2, y_center1 + h1 / 2
    x3, y3, x4, y4 = x_center2 - w2 / 2, y_center2 - h2 / 2, x_center2 + w2 / 2, y_center2 + h2 / 2

    intersection = max(0, min(x2, x4) - max(x1, x3)) * max(0, min(y2, y4) - max(y1, y3))
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x4 - x3) * (y4 - y3)

    iou = intersection / (area1 + area2 - intersection)
    return iou > threshold


def augment_copy_paste(images_path, labels_path):
    image_files = [f for f in os.listdir(images_path) if f.endswith(('.jpg', '.png'))]

    for image_file in image_files:
        print(image_file)
        has_person = True
        img = cv2.imread(os.path.join(images_path, image_file))
        label_file = os.path.join(labels_path, os.path.splitext(image_file)[0] + '.txt')
        current_boxes = get_boxes_from_label(label_file)

        donor_image_file = random.choice(image_files)
        while donor_image_file == image_file:
            donor_image_file = random.choice(image_files)

        donor_img = cv2.imread(os.path.join(images_path, donor_image_file))
        donor_label_file = os.path.join(labels_path, os.path.splitext(donor_image_file)[0] + '.txt')
        donor_boxes = get_boxes_from_label(donor_label_file)
        while has_person:
            label_file = donor_label_file
            with open(label_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if int(line.strip().split()[0]) == 0:
                        has_person =False
            if has_person:
                donor_image_file = random.choice(image_files)
                while donor_image_file == image_file:
                    donor_image_file = random.choice(image_files)

                donor_img = cv2.imread(os.path.join(images_path, donor_image_file))
                donor_label_file = os.path.join(labels_path, os.path.splitext(donor_image_file)[0] + '.txt')
                donor_boxes = get_boxes_from_label(donor_label_file)
        for box in donor_boxes:
            overlap = any([check_overlap(box, current_box) for current_box in current_boxes])
            if not overlap:
                paste_image(donor_img, img, box)
                current_boxes.append(box)
        if not os.path.exists(augmented_img_path):
            os.makedirs(augmented_img_path)
        if not os.path.exists(augmented_txt_path):
            os.makedirs(augmented_txt_path)
        cv2.imwrite(os.path.join(augmented_img_path, "augmented_" + image_file), img)

        with open(os.path.join(augmented_txt_path, "augmented_" + os.path.splitext(image_file)[0] + '.txt'), 'w') as f:
            for box in current_boxes:
                f.write("0 " + " ".join(map(str, box)) + "\n")


images_path = r"E:\BaiduNetdiskDownload\KT5\images\val"
labels_path = r"E:\BaiduNetdiskDownload\KT5\labels\val"
augmented_img_path = r"E:\BaiduNetdiskDownload\KT5\augimages\val"
augmented_txt_path = r"E:\BaiduNetdiskDownload\KT5\auglabels\val"
augment_copy_paste(images_path, labels_path)
