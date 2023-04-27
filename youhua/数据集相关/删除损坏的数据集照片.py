import os
from PIL import Image
import cv2
def is_image_valid(file_path):
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except (IOError, SyntaxError):
        return False

def delete_corrupted_images_and_labels(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            file_path = os.path.join(directory, filename)
            im = cv2.imread(file_path)  # BGR
            if im is None:
                print(f"Deleting corrupted image: {filename}")
                os.remove(file_path)

                label_filename = filename.replace(".jpg", ".txt")
                label_file_path = os.path.join(directory, label_filename)
                if os.path.exists(label_file_path):
                    print(f"Deleting corresponding label file: {label_filename}")
                    # os.remove(label_file_path)
                else:
                    print(f"Label file not found: {label_filename}")

# 要处理的目录
directory_path = r"D:\xian_yu\VOCdevkit\images\train"
delete_corrupted_images_and_labels(directory_path)
