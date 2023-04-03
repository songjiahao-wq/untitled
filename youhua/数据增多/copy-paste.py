import os
import random
import cv2
import xml.etree.ElementTree as ET

def get_bboxes_from_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    bboxes = []
    for obj in root.findall("object"):
        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)
        bboxes.append((xmin, ymin, xmax, ymax))
    return bboxes


def create_copypaste_image(img1, img2, bboxes, min_size=50):
    img1_copy = img1.copy()
    h1, w1, _ = img1.shape
    h2, w2, _ = img2.shape

    for bbox in bboxes:
        xmin, ymin, xmax, ymax = bbox

        # 检查边界是否合法
        if xmax <= w1 and ymax <= h1:
            # 裁剪图像2中的物体
            patch = img2[ymin:ymax, xmin:xmax]
            patch_h, patch_w, _ = patch.shape

            # 如果物体的尺寸小于最小尺寸，则调整其大小
            if patch_h < min_size or patch_w < min_size:
                scale_factor = max(min_size / patch_h, min_size / patch_w)
                new_h, new_w = int(patch_h * scale_factor), int(patch_w * scale_factor)
                patch = cv2.resize(patch, (new_w, new_h))

                # 更新边界框大小以适应新的尺寸
                xmax = xmin + new_w
                ymax = ymin + new_h

                # 如果新的边界框超出图像1的边界，则跳过此物体
                if xmax > w1 or ymax > h1:
                    continue

            # 计算目标区域的宽度和高度
            target_h = ymax - ymin
            target_w = xmax - xmin

            # 如果目标区域的宽度或高度与物体的宽度或高度不匹配，调整物体的大小
            if patch_h != target_h or patch_w != target_w:
                patch = cv2.resize(patch, (target_w, target_h))

            img1_copy[ymin:ymax, xmin:xmax] = patch

    return img1_copy


def copypaste_data_augmentation_single(image_folder, annotation_folder, output_folder,rate = 0.5):
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, "JPEGImages"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "Annotations"), exist_ok=True)

    image_files = [f for f in os.listdir(image_folder) if f.endswith(".jpg")]
    annotation_files = [f for f in os.listdir(annotation_folder) if f.endswith(".xml")]

    for i, img1_name in enumerate(image_files):
        img2_name = random.choice(image_files)

        while img1_name == img2_name:
            img2_name = random.choice(image_files)

        img1_path = os.path.join(image_folder, img1_name)
        img2_path = os.path.join(image_folder, img2_name)
        xml1_path = os.path.join(annotation_folder, img1_name.replace(".jpg", ".xml"))
        xml2_path = os.path.join(annotation_folder, img2_name.replace(".jpg", ".xml"))

        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        bboxes1 = get_bboxes_from_xml(xml1_path)
        bboxes2 = get_bboxes_from_xml(xml2_path)

        copypaste_img = create_copypaste_image(img1, img2, bboxes2)

        output_img_path = os.path.join(output_folder, "JPEGImages", f"copypaste_{i}.jpg")
        cv2.imwrite(output_img_path, copypaste_img)

        output_xml_path = os.path.join(output_folder, "Annotations", f"copypaste_{i}.xml")
        tree = ET.parse(xml1_path)
        tree.write(output_xml_path)





# 示例：
image_folder = r"D:\songjiahao\DATA\smokke\VOC\images"
annotation_folder = r"D:\songjiahao\DATA\smokke\VOC\Annotions"
output_folder = r"D:\songjiahao\DATA\smokke\VOC\out"

copypaste_data_augmentation_single(image_folder, annotation_folder, output_folder,rate = 0.5)



