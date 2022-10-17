"""这个脚本从Annotations中随机划分训练集和测试集，最终生成ImagesSet/train.txt和val.txt"""
import os
import random
from os.path import *


# --------------------------全局地址变量--------------------------------#
dir_path = dirname(dirname(abspath(__file__)))  # 当前项目位置
xml_path = os.path.join(dir_path, "Pest", "Annotations")  # xml文件位置
assert os.path.exists(xml_path), "xml_path not exist!"

# 建立ImageSets文件
ImageSets_path = os.path.join(dir_path, "Pest", "ImageSets")
if not os.path.exists(ImageSets_path):
    os.makedirs(ImageSets_path)

# train.txt 和 val.txt文件位置
traintxt_path = os.path.join(dir_path, "Pest", "ImageSets", "train.txt")
valtxt_path = os.path.join(dir_path, "Pest", "ImageSets", "val.txt")
testtxt_path = os.path.join(dir_path, "Pest", "ImageSets", "test.txt")

# 如果已经存在txt文件 则删除
if os.path.exists(traintxt_path):
    os.remove(traintxt_path)
if os.path.exists(valtxt_path):
    os.remove(valtxt_path)
if os.path.exists(testtxt_path):
    os.remove(testtxt_path)
# --------------------------全局地址变量--------------------------------#
def create_imagesets_train_val(xml_full_path, traintxt_full_path, valtxt_full_path):
    # 训练集比例
    train_percent = 0.8
    # 验证集比例
    val_percent = 0.2
    # xml文件目录位置
    xml_path = xml_full_path
    total_xml = os.listdir(xml_path)  # 获得目录下所有xml文件

    num = len(total_xml)
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
        name = total_xml[i][:-4] + '\n'
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

if __name__ == '__main__':
    create_imagesets_train_val_test(xml_path, traintxt_path, valtxt_path, testtxt_path)
