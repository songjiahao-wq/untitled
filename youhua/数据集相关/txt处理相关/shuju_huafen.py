# 将图片和标注数据按比例切分为 训练集和测试集
import shutil
import random
import os
import tqdm

# 原始路径
image_original_path = r'E:\BaiduNetdiskDownload\crow/'
label_original_path = r'E:\BaiduNetdiskDownload\crow/'
# 复制文件总路径
src_root = r'E:\BaiduNetdiskDownload\Crowdhuman\yolo'
# 训练集路径
train_image_path = src_root + '/images/train/'
train_label_path = src_root + '/labels/train/'
# 验证集路径
val_image_path = src_root + '/images/val/'
val_label_path = src_root + '/labels/val/'
# 测试集路径
test_image_path = src_root + '/images/test/'
test_label_path = src_root + '/labels/test/'

# 数据集划分比例，训练集75%，验证集15%，测试集15%
train_percent = 0.85
val_percent = 0.15
test_percent = 0.0
# 数据集划分数量，训练集800，验证集800，测试集800
# num_train = 850
# num_val = 800
# num_test = 5000

# 检查文件夹是否存在
def mkdir():
    if not os.path.exists(train_image_path):
        os.makedirs(train_image_path)
    if not os.path.exists(train_label_path):
        os.makedirs(train_label_path)

    if not os.path.exists(val_image_path):
        os.makedirs(val_image_path)
    if not os.path.exists(val_label_path):
        os.makedirs(val_label_path)

    if not os.path.exists(test_image_path):
        os.makedirs(test_image_path)
    if not os.path.exists(test_label_path):
        os.makedirs(test_label_path)


def main():
    mkdir()

    total_txt = os.listdir(label_original_path)
    num_txt = len(total_txt)
    list_all_txt = list(range(num_txt))  # 范围 range(0, num)

    if train_percent:
        num_train = int(num_txt * train_percent)
        num_val = int(num_txt * val_percent)
        num_test = num_txt - num_train - num_val

    # 随机选择训练集
    train = random.sample(list_all_txt, num_train)
    # 从剩下的列表中选择验证集
    remaining = [i for i in list_all_txt if i not in train]
    val = random.sample(remaining, num_val)
    # 剩下的就是测试集
    test = [i for i in remaining if i not in val]

    # 检查两个列表元素是否有重合的元素
    assert len(set(train) & set(val)) == 0, "Train and Validation sets have overlapping files!"
    assert len(set(train) & set(test)) == 0, "Train and Test sets have overlapping files!"
    assert len(set(val) & set(test)) == 0, "Validation and Test sets have overlapping files!"

    # 复制文件
    for i in tqdm.tqdm(list_all_txt):
        name = total_txt[i][:-4]
        srcImage = image_original_path + name + '.jpg'
        srcLabel = label_original_path + name + '.txt'

        if i in train:
            dst_train_Image = train_image_path + name + '.jpg'
            dst_train_Label = train_label_path + name + '.txt'
            shutil.copyfile(srcImage, dst_train_Image)
            shutil.copyfile(srcLabel, dst_train_Label)
        elif i in val:
            dst_val_Image = val_image_path + name + '.jpg'
            dst_val_Label = val_label_path + name + '.txt'
            shutil.copyfile(srcImage, dst_val_Image)
            shutil.copyfile(srcLabel, dst_val_Label)
        elif i in test:
            dst_test_Image = test_image_path + name + '.jpg'
            dst_test_Label = test_label_path + name + '.txt'
            shutil.copyfile(srcImage, dst_test_Image)
            shutil.copyfile(srcLabel, dst_test_Label)

if __name__ == '__main__':
    main()


