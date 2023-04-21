# pip install albumentations
'''
使用了一些常见的数据增强技术，如水平翻转、随机亮度对比度、平移缩放旋转、高斯噪声和粗略丢失。
可以根据需要自定义transform管道中的数据增强技术。
'''
import os
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 设置输入输出文件夹
input_image_folder = 'path/to/input/images'
input_label_folder = 'path/to/input/labels'
output_image_folder = 'path/to/output/images'
output_label_folder = 'path/to/output/labels'

# 数据增强管道
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.ShiftScaleRotate(scale_limit=0.1, rotate_limit=20, p=0.3),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    A.CoarseDropout(max_holes=5, p=0.3),
    A.Normalize(),
    ToTensorV2(),
])

# 遍历图像和标签文件夹
image_files = os.listdir(input_image_folder)
label_files = os.listdir(input_label_folder)

for image_file, label_file in zip(image_files, label_files):
    image_path = os.path.join(input_image_folder, image_file)
    label_path = os.path.join(input_label_folder, label_file)

    image = cv2.imread(image_path)
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

    # 数据增强
    augmented = transform(image=image, mask=label)
    image_aug = augmented['image']
    label_aug = augmented['mask']

    # 保存增强后的图像和标签
    output_image_path = os.path.join(output_image_folder, image_file)
    output_label_path = os.path.join(output_label_folder, label_file)

    cv2.imwrite(output_image_path, image_aug)
    cv2.imwrite(output_label_path, label_aug)
