"""
功能
1.根据train.txt和val.txt将voc数据集标注信息(.xml)转为yolo标注格式(.txt)，生成dataset文件（train+val）
"""
import os
from tqdm import tqdm
from lxml import etree
import json
import shutil
from os.path import *

# --------------------------全局地址变量--------------------------------#
# 拼接出voc的images目录，xml目录，txt目录
dir_path = dirname(dirname(abspath(__file__)))  # 'E:\\yolov3_spp\\data' 当前数据集的位置
images_path = os.path.join(dir_path, "Datasets", "Images")  # 数据集图片目录
xml_path = os.path.join(dir_path, "Datasets", "Annotations")  # 数据集xml文件位置
train_txt_path = os.path.join(dir_path, "Datasets", "ImageSets", "train.txt")  # 数据集ImageSets中train位置
val_txt_path = os.path.join(dir_path, "Datasets", "ImageSets", "val.txt")  # 数据集ImageSets中val位置
test_txt_path = os.path.join(dir_path, "Datasets", "ImageSets", "test.txt")  # 数据集ImageSets中test位置
label_json_path = os.path.join(dir_path, "pest_classes.json")  # 数据集label标签对应json文件
save_file_root = os.path.join(dir_path, "pest")  # 新数据集的根目录

# 检查文件/文件夹都是否存在
assert os.path.exists(images_path), "images path not exist..."
assert os.path.exists(xml_path), "xml path not exist..."
assert os.path.exists(train_txt_path), "train txt file not exist..."
assert os.path.exists(val_txt_path), "val txt file not exist..."
assert os.path.exists(test_txt_path), "test txt file not exist..."
assert os.path.exists(label_json_path), "label_json_path does not exist..."

if os.path.exists(save_file_root) is False:  # 没有根目录就新建
    os.makedirs(save_file_root)
# --------------------------全局地址变量--------------------------------#
def parse_xml_to_dict(xml):
    """
    将xml文件解析成字典形式，参考tensorflow的recursive_parse_xml_to_dict
    Args：
        xml: xml tree obtained by parsing XML file contents using lxml.etree
    Returns:
        Python dictionary holding XML contents.
    """
    if len(xml) == 0:  # 遍历到底层，直接返回tag对应的信息
        return {xml.tag: xml.text}

    result = {}
    for child in xml:
        child_result = parse_xml_to_dict(child)  # 递归遍历标签信息
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:  # 因为object可能有多个，所以需要放入列表里
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}

def translate_info(file_names: list, save_root: str, class_dict: dict, train_val_test='train'):
    """
    将对应xml文件信息转为yolo中使用的txt文件信息  xml to txt  xyxy to xywh(normalization)
    :param file_names: train/val/test的所有图片名 如：['20210819B000001', '20210819B000002', '20210819B000004',...]
    :param save_root: 新数据集的root目录
    :param class_dict: 新数据集的label字典 如：{'powdery_mildew': 0, 'leaf_miner': 1, 'anthracnose': 2}
    :param train_val_test: 是什么数据 train or val or test
    :return:
    """
    save_txt_path = os.path.join(save_root, train_val_test, "labels")  # 新数据集label目录
    if os.path.exists(save_txt_path) is False:  # 没有就新建
        os.makedirs(save_txt_path)
    save_images_path = os.path.join(save_root, train_val_test, "images")  # 新数据集images目录
    if os.path.exists(save_images_path) is False:  # 没有就新建
        os.makedirs(save_images_path)

    # 遍历train/val/test中的所有文件名
    for file in tqdm(file_names, desc="translate {} file...".format(train_val_test)):
        # 检查下图像文件是否存在
        img_path = os.path.join(images_path, file + ".jpg")  # 当前图像路径
        assert os.path.exists(img_path), "file:{} not exist...".format(img_path)

        # 检查xml文件是否存在
        xml_full_path = os.path.join(xml_path, file + ".xml")  # 当前xml路径
        assert os.path.exists(xml_full_path), "file:{} not exist...".format(xml_full_path)

        # read xml
        with open(xml_full_path, encoding='UTF-8') as fid:
            xml_str = fid.read()  # 读取xml数据
        xml = etree.fromstring(xml_str)
        data = parse_xml_to_dict(xml)["annotation"]  # 读取xml中annotation标签的内容 xml to dict
        img_height = int(data["size"]["height"])  # 读取xml中图片height
        img_width = int(data["size"]["width"])  # 读取xml中图片width

        # 将当前xml文件中（annotation）的object信息转为txt格式
        with open(os.path.join(save_txt_path, file + ".txt"), "w") as f:
            assert "object" in data.keys(), "file: '{}' lack of object key.".format(xml_full_path)
            for index, obj in enumerate(data["object"]):
                # 获取每个object的box信息
                xmin = float(obj["bndbox"]["xmin"])
                xmax = float(obj["bndbox"]["xmax"])
                ymin = float(obj["bndbox"]["ymin"])
                ymax = float(obj["bndbox"]["ymax"])
                class_name = obj["name"].strip()
                class_index = class_dict[class_name]    # object id从0开始

                # 将box信息转换到yolo格式  xyxy to xywh
                xcenter = xmin + (xmax - xmin) / 2
                ycenter = ymin + (ymax - ymin) / 2
                w = xmax - xmin
                h = ymax - ymin

                # 绝对坐标转相对坐标，保存6位小数  归一化
                xcenter = round(xcenter / img_width, 6)
                ycenter = round(ycenter / img_height, 6)
                w = round(w / img_width, 6)
                h = round(h / img_height, 6)

                info = [str(i) for i in [class_index, xcenter, ycenter, w, h]]

                # 写入当前txt文件中
                if index == 0:
                    f.write(" ".join(info))
                else:
                    f.write("\n" + " ".join(info))

        # copy image into save_images_path
        shutil.copyfile(img_path, os.path.join(save_images_path, img_path.split(os.sep)[-1]))

if __name__ == "__main__":
    # read class_indict 先取得数据集对应的标签dict
    json_file = open(label_json_path, 'r')
    class_dict = json.load(json_file)

    # 读取train.txt中的所有行信息，删除空行
    with open(train_txt_path, "r") as r:
        train_file_names = [i for i in r.read().splitlines() if len(i.strip()) > 0]
    # voc信息转yolo(xml to txt)，并将图像文件复制到相应文件夹
    translate_info(train_file_names, save_file_root, class_dict, "train")

    # 读取val.txt中的所有行信息，删除空行
    with open(val_txt_path, "r") as r:
        val_file_names = [i for i in r.read().splitlines() if len(i.strip()) > 0]
    # voc信息转yolo(xml to txt)，并将图像文件复制到相应文件夹
    translate_info(val_file_names, save_file_root, class_dict, "val")

    # 读取test.txt中的所有行信息，删除空行
    with open(test_txt_path, "r") as r:
        test_file_names = [i for i in r.read().splitlines() if len(i.strip()) > 0]
    # voc信息转yolo(xml to txt)，并将图像文件复制到相应文件夹
    translate_info(test_file_names, save_file_root, class_dict, "test")

