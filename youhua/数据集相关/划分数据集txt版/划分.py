import os
import random
import shutil


# --------------------------全局地址变量--------------------------------#
master_root = os.path.abspath(os.path.join(os.getcwd(), "../../"))  # 'I:\\hive-master'
data_root = os.path.join(master_root, "datasets")  # 'I:\\hive-master\\datasets'
image_full_path = os.path.join(data_root, "hive", "images")
label_full_path = os.path.join(data_root, "hive", "labels")  # 'I:\\hive-master\\datasets\\hive\\labels'


# 建立ImageSets文件
ImageSets_path = os.path.join(data_root, "hive", "ImageSets")
if not os.path.exists(ImageSets_path):
    os.makedirs(ImageSets_path)

# train.txt 和 val.txt文件位置
traintxt_path = os.path.join(data_root, "hive", "ImageSets", "train.txt")
valtxt_path = os.path.join(data_root, "hive", "ImageSets", "val.txt")
testtxt_path = os.path.join(data_root, "hive", "ImageSets", "test.txt")

# 如果已经存在txt文件 则删除
if os.path.exists(traintxt_path):
    os.remove(traintxt_path)
if os.path.exists(valtxt_path):
    os.remove(valtxt_path)
if os.path.exists(testtxt_path):
    os.remove(testtxt_path)
# --------------------------全局地址变量--------------------------------#

def create_imagesets_train_val(label_full_path, traintxt_full_path, valtxt_full_path):
    # 训练集比例
    train_percent = 0.8
    # 验证集比例
    val_percent = 0.2
    # label文件目录位置
    label_path = label_full_path
    total_label = os.listdir(label_path)  # 获得目录下所有xml文件

    num = len(total_label)
    lists = list(range(num))

    num_train = int(num * train_percent)

    # 随机选num_train个train文件
    train_list = random.sample(lists, num_train)
    for i in train_list:
        lists.remove(i)
    val_list = lists  # val等于train剩下的 这里没有划分test

    ftrain = open(traintxt_full_path, 'w')
    fval = open(valtxt_full_path, 'w')

    for i in range(num):
        name = total_label[i][:-4] + '\n'  # 出去.xml或.txt
        if i in train_list:
            ftrain.write(name)  # train.txt文件写入
        else:
            fval.write(name)  # val.txt文件写入

    ftrain.close()  # 关闭train.txt
    fval.close()    # 关闭val.txt

def create_imagesets_train_val_test(xml_full_path, traintxt_full_path, valtxt_full_path, testtxt_full_path):
    # 训练集比例
    train_percent = 0.6
    # 验证集比例
    val_percent = 0.2
    # 测试集比例
    test_percent = 0.2
    # xml文件目录位置
    xml_path = xml_full_path
    total_xml = os.listdir(xml_path)  # 获得目录下所有xml文件

    num = len(total_xml)
    lists = list(range(num))

    num_train = int(num * train_percent)  # 训练集个数

    num_val = int(num * val_percent)  # 验证集个数

    # 随机选num_train个train文件
    train_list = random.sample(lists, num_train)
    for i in train_list:
        lists.remove(i)
    val_list = random.sample(lists, num_val)
    for j in val_list:
        lists.remove(j)
    test_list = lists

    ftrain = open(traintxt_full_path, 'w')
    fval = open(valtxt_full_path, 'w')
    ftest = open(testtxt_full_path, 'w')

    for i in range(num):
        name = total_xml[i][:-4] + '\n'
        if i in train_list:
            ftrain.write(name)  # train.txt文件写入
        elif i in val_list:
            fval.write(name)  # val.txt文件写入
        else:
            ftest.write(name)  # test.txt文件写入

    ftrain.close()  # 关闭train.txt
    fval.close()  # 关闭val.txt
    ftest.close()  # 关闭test.txt

def split_image_label_train_val(image_full_path, label_full_path, traintxt_path, valtxt_path):
    image_train_path = os.path.join(image_full_path, "train")
    image_val_path = os.path.join(image_full_path, "val")
    label_train_path = os.path.join(label_full_path, "train")
    label_val_path = os.path.join(label_full_path, "val")
    if not os.path.exists(image_train_path):
        os.makedirs(image_train_path)
    if not os.path.exists(image_val_path):
        os.makedirs(image_val_path)
    if not os.path.exists(label_train_path):
        os.makedirs(label_train_path)
    if not os.path.exists(label_val_path):
        os.makedirs(label_val_path)
    ftraintxt = open(traintxt_path, 'r')
    for line in ftraintxt:
        train_image_name = line.split('\n')[0]
        train_image_full_name = train_image_name + ".jpg"
        train_label_full_name = train_image_name + '.txt'
        train_image_full_path = os.path.join(image_full_path, train_image_full_name)
        train_label_full_path = os.path.join(label_full_path, train_label_full_name)
        shutil.copy(train_image_full_path, image_train_path)
        shutil.copy(train_label_full_path, label_train_path)
    ftraintxt.close()

    fvaltxt = open(valtxt_path, 'r')
    for line in fvaltxt:
        val_image_name = line.split('\n')[0]
        val_image_full_name = val_image_name + ".jpg"
        val_label_full_name = val_image_name + '.txt'
        val_image_full_path = os.path.join(image_full_path, val_image_full_name)
        val_label_full_path = os.path.join(label_full_path, val_label_full_name)
        shutil.copy(val_image_full_path, image_val_path)
        shutil.copy(val_label_full_path, label_val_path)
    fvaltxt.close()


if __name__ == '__main__':
    # 1、划分train.txt和val.txt
    create_imagesets_train_val(label_full_path, traintxt_path, valtxt_path)
    # create_imagesets_train_val_test(label_full_path, traintxt_path, valtxt_path, testtxt_path)

    # 2、按照train.txt和val.txt 将所有images和txt文件划分为train和val两部分
    split_image_label_train_val(image_full_path, label_full_path, traintxt_path, valtxt_path)



