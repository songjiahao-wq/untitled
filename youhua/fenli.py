# 将图片和标注数据按比例切分为 训练集和测试集
import shutil
import random
import os

# 原始路径
image_original_path = r'D:\songjiahao\DATA\crowhuman\trainJPEGImages/'
label_original_path = r'D:\songjiahao\DATA\crowhuman\trainAnnotations/'
# 训练集路径
train_image_path = r'D:\songjiahao\DATA\crowhuman\trainIma/'
train_label_path = r'D:\songjiahao\DATA\crowhuman\trainAnno/'
# 验证集路径
val_image_path = r'D:\songjiahao\DATA\crowhuman\valJPEGImages/'
val_label_path = r'D:\songjiahao\DATA\crowhuman\valAnnotations/'
# 测试集路径
test_image_path = r'D:\songjiahao\DATA\NEU\trainNEU\test/'
test_label_path = r'D:\songjiahao\DATA\NEU\trainNEU\test/'

# 数据集划分比例，训练集75%，验证集15%，测试集15%
train_percent = 0.9
val_percent = 0.1
# test_percent = 0.1


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
    # mkdir()

    total_txt = os.listdir(label_original_path)
    num_txt = len(total_txt)
    list_all_txt = range(num_txt)  # 范围 range(0, num)

    num_train = int(num_txt * train_percent)
    num_val = int(num_txt * val_percent)
    num_test = num_txt - num_train - num_val

    train = random.sample(list_all_txt, num_train)
    # train从list_all_txt取出num_train个元素
    # 所以list_all_txt列表只剩下了这些元素：val_test
    val_test = [i for i in list_all_txt if not i in train]
    # 再从val_test取出num_val个元素，val_test剩下的元素就是test
    val = random.sample(val_test, num_val)
    # 检查两个列表元素是否有重合的元素
    # set_c = set(val_test) & set(val)
    # list_c = list(set_c)
    # print(list_c)
    # print(len(list_c))

    print("训练集数目：{}, 验证集数目：{},测试集数目：{}".format(len(train), len(val), len(val_test) - len(val)))
    for i in list_all_txt:
        name = total_txt[i][:-4]

        srcImage = image_original_path + name + '.jpg'
        srcLabel = label_original_path + name + '.txt'

        if i in train:
            dst_train_Image = train_image_path + name + '.jpg'
            dst_train_Label = train_label_path + name + '.txt'
            shutil.copyfile(srcImage, dst_train_Image)
            shutil.copyfile(srcLabel, dst_train_Label)
        if i in val:
            dst_val_Image = val_image_path + name + '.jpg'
            dst_val_Label = val_label_path + name + '.txt'
            shutil.copyfile(srcImage, dst_val_Image)
            shutil.copyfile(srcLabel, dst_val_Label)
        # else:
        #     dst_test_Image = test_image_path + name + '.jpg'
        #     dst_test_Label = test_label_path + name + '.txt'
        #     shutil.copyfile(srcImage, dst_test_Image)
        #     shutil.copyfile(srcLabel, dst_test_Label)


if __name__ == '__main__':
    main()


