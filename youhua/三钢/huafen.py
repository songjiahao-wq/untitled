# 随机分成三份
import shutil
import random
import os

image_original_path = r'D:\app\PotPlayer\Capture\quanbu/'

image_now_path1 = r'D:\app\PotPlayer\Capture\fenli1/'
image_now_path2 = r'D:\app\PotPlayer\Capture\fenli2/'
image_now_path3 = r'D:\app\PotPlayer\Capture\fenli3/'

# 数据集划分比例，训练集75%，验证集15%，测试集15%
path1_percent = 0.33
path2_percent = 0.33
path3_percent = 0.33
# 检查文件夹是否存在
def mkdir():
    if not os.path.exists(image_now_path1):
        os.makedirs(image_now_path1)
    if not os.path.exists(image_now_path1):
        os.makedirs(image_now_path1)

    if not os.path.exists(image_now_path2):
        os.makedirs(image_now_path2)
    if not os.path.exists(image_now_path3):
        os.makedirs(image_now_path3)


def main():
    mkdir()

    total_txt = os.listdir(image_original_path)
    num_txt = len(total_txt)
    list_all_txt = range(num_txt)  # 范围 range(0, num)

    num1 = int(num_txt * path1_percent)
    num2 = int(num_txt * path1_percent)
    num3 = num_txt - num1 - num2

    num_1 = random.sample(list_all_txt, num1)
    num_3 = [i for i in list_all_txt if not i in num_1]
    num_2 = random.sample(num_3, num2)

    print("训练集数目：{}, 验证集数目：{},测试集数目：{}".format(len(num_1), len(num_2), len(num_3)
                                               -len(num_2)))
    for i in list_all_txt:
        name = total_txt[i][:-4]
        srcImage = image_original_path + name + '.jpg'

        if i in num_1:
            dst_train_Image = image_now_path1 + name + '.jpg'
            shutil.copyfile(srcImage, dst_train_Image)
        elif i in num_2:
            dst_val_Image = image_now_path2 + name + '.jpg'
            shutil.copyfile(srcImage, dst_val_Image)
        else:
            dst_test_Image = image_now_path3 + name + '.jpg'
            shutil.copyfile(srcImage, dst_test_Image)



if __name__ == '__main__':
    main()