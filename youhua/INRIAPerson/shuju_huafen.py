# 将图片和标注数据按比例切分为 训练集和测试集
import shutil
import random
import os

# 原始路径
image_original_path = r'E:\BaiduNetdiskDownload\Crowdhuman\images/'
label_original_path = r'E:\BaiduNetdiskDownload\Crowdhuman\labels/'
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
    list_all_txt = list(range(num_txt))  # 将范围对象转换为列表

    num_train = int(num_txt * train_percent)
    num_val = int(num_txt * val_percent)
    num_test = num_txt - num_train - num_val

    # 训练集划分
    train = random.sample(list_all_txt, num_train)
    remaining = [i for i in list_all_txt if i not in train]

    # 验证集划分
    val = random.sample(remaining, num_val)

    # 剩余数据作为测试集
    test = [i for i in remaining if i not in val]

    print(f"Train set size: {len(train)}, Validation set size: {len(val)}, Test set size: {len(test)}")


    # print("训练集数目：{}, 验证集数目：{},测试集数目：{}".format(len(train), len(val), len(val_test) - len(val)))
    for i in list_all_txt:
        name = total_txt[i][:-4]

        srcImage = image_original_path + name + '.jpg'
        srcLabel = label_original_path + name + '.txt'

        if i in train:
            if os.path.exists(srcImage):
                dst_train_Image = train_image_path + name + '.jpg'
                dst_train_Label = train_label_path + name + '.txt'
                shutil.copyfile(srcImage, dst_train_Image)
                shutil.copyfile(srcLabel, dst_train_Label)
            else:
                print(srcImage)
        elif i in val:
            if os.path.exists(srcImage):
                dst_val_Image = val_image_path + name + '.jpg'
                dst_val_Label = val_label_path + name + '.txt'
                shutil.copyfile(srcImage, dst_val_Image)
                shutil.copyfile(srcLabel, dst_val_Label)
            else:
                print(srcImage)
        else:
            dst_test_Image = test_image_path + name + '.jpg'
            dst_test_Label = test_label_path + name + '.txt'
            shutil.copyfile(srcImage, dst_test_Image)
            shutil.copyfile(srcLabel, dst_test_Label)


if __name__ == '__main__':
    main()


