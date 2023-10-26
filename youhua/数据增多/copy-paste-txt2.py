import os
import random
import cv2
import numpy as np
"""
现在有yolo 的images文件和labels文件，分别存放图像和标签文件，请实现copy-paste数据增强代码，要求：可以将标签为0的行人目标从当前图中提取出来，
随机粘贴到当前图像中，并得到增强后的标签，并且粘贴的目标不能与当前图像中已存在的所有目标重合太大。流程为：先提取当前图像中所有标签为0的行人目标，
然后从已提取的标签中随机抽取三个目标，粘贴到当前图像中，保存增强的图像和标签文件
"""
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


def paste_person(src_img, target_img, src_box, existing_boxes, threshold=0.3):
    """
    Paste src_box from src_img to target_img ensuring no significant overlap.
    """
    H, W, _ = target_img.shape
    h, w, _ = src_box.shape
    max_attempts = 10  # We try 10 times to find a non-overlapping position

    for _ in range(max_attempts):
        x = random.randint(0, W - w)
        y = random.randint(0, H - h)
        overlap = False
        for box in existing_boxes:
            if iou([x, y, x + w, y + h], box) > threshold:
                overlap = True
                break
        if not overlap:
            target_img[y:y + h, x:x + w] = src_box
            return target_img, [x, y, x + w, y + h]
    return target_img, None


def augment_single_image(img_path, label_path):
    """
    Augment a single image by copy-pasting boxes.
    """
    img = cv2.imread(img_path)
    with open(label_path, 'r') as f:
        lines = f.readlines()
        person_boxes = [list(map(float, line.strip().split()[1:4])) for line in lines if
                        int(line.strip().split()[0]) == 0]

    # Randomly choose 3 boxes
    chosen_boxes = random.sample(person_boxes, min(3, len(person_boxes)))

    new_boxes = []
    for box in chosen_boxes:
        x, y, w, h = box
        src_box = img[int(y):int(y + h), int(x):int(x + w)]
        img, new_box = paste_person(src_box, img, box, person_boxes)
        if new_box:
            new_boxes.append(new_box)

    # Save augmented image
    cv2.imwrite(img_path, img)

    # Update labels
    with open(label_path, 'a') as f:
        for box in new_boxes:
            x, y, x2, y2 = box
            w = x2 - x
            h = y2 - y
            f.write(f"0 {x + w / 2} {y + h / 2} {w} {h}\n")


# Usage
labels_dir = './labels'
images_dir = './images'

for img_file in os.listdir(images_dir):
    img_path = os.path.join(images_dir, img_file)
    label_path = os.path.join(labels_dir, img_file.replace('.jpg', '.txt'))
    augment_single_image(img_path, label_path)
"""
上述代码首先定义了计算IoU（Intersection over Union）的函数，这可以帮助我们判断两个框之间的重叠程度。然后，定义了一个函数来粘贴一个目标到一个
图像上，并确保它与现有的框没有太大的重叠。最后，对每张图片进行增强操作。这只是一个基于您需求的简化版本的数据增强流程，可能需要进一步的优化来适应您的
具体情况。
"""