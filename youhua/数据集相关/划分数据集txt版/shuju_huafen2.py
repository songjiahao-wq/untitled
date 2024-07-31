# -*- coding: utf-8 -*-
# @Time    : 2024/7/29 9:22
# @Author  : sjh
# @Site    :
# @File    : 划分数据集.py
# @Comment :
import os
import random
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool, Manager


def create_imagesets_train_val(images_dir, label_dir, train_file, val_file, train_percent=0.8, val_percent=0.2):
    total_labels = os.listdir(label_dir)
    total_labels = [label for label in total_labels if label.endswith('.txt')]
    num_labels = len(total_labels)
    num_train = int(num_labels * train_percent)

    random.shuffle(total_labels)

    train_labels = total_labels[:num_train]
    val_labels = total_labels[num_train:]

    with open(train_file, 'w') as ftrain, open(val_file, 'w') as fval:
        for label in train_labels:
            label = Path(label)
            ftrain.write(label.stem + '\n')
        for label in val_labels:
            fval.write(label.stem + '\n')


def os_walk_find_images(txtpath, images_dir):
    """
    查找与txt对应名称的照片
    """
    for root, dirs, files in os.walk(images_dir):
        for file in files:
            if Path(file).stem == Path(txtpath).stem:
                yield os.path.join(root, file)


def write_paths(label, images_dir, queue):
    paths = list(os_walk_find_images(label, images_dir))
    queue.put(len(paths))
    return paths


def create_imagesets_train_val_test(images_dir, label_dir, train_file, val_file, test_file, train_percent=0.6,
                                    val_percent=0.2):
    total_labels = os.listdir(label_dir)
    num_labels = len(total_labels)
    num_train = int(num_labels * train_percent)
    num_val = int(num_labels * val_percent)

    random.shuffle(total_labels)

    train_labels = total_labels[:num_train]
    val_labels = total_labels[num_train:num_train + num_val]
    test_labels = total_labels[num_train + num_val:]

    def process_labels(file, labels, images_dir):
        with open(file, 'w') as f:
            with Manager() as manager:
                queue = manager.Queue()
                with Pool() as pool:
                    results = pool.starmap_async(write_paths, [(label, images_dir, queue) for label in labels])

                    pbar = tqdm(total=len(labels))
                    while not results.ready():
                        pbar.update(queue.qsize())
                        while not queue.empty():
                            queue.get()
                    pbar.close()

                    for image_paths in results.get():
                        for image_path in image_paths:
                            f.write(image_path + '\n')

    process_labels(train_file, train_labels, images_dir)
    process_labels(val_file, val_labels, images_dir)
    process_labels(test_file, test_labels, images_dir)


if __name__ == '__main__':
    # 指定路径
    labels_dir = r"/home/ada/sjh/datasets/DensePose/labels/val2014"
    images_dir = r"/home/ada/sjh/datasets/DensePose/images/val2014"
    save_dir = r"/home/ada/sjh/datasets/DensePose/"

    # 创建ImageSets文件夹和txt文件路径
    imagesets_dir = os.path.join(save_dir, "ImageSets")
    os.makedirs(imagesets_dir, exist_ok=True)

    train_file = os.path.join(imagesets_dir, "train.txt")
    val_file = os.path.join(imagesets_dir, "val.txt")
    test_file = os.path.join(imagesets_dir, "test.txt")

    # 划分数据集
    # create_imagesets_train_val_test(images_dir, labels_dir, train_file, val_file)
    create_imagesets_train_val_test(images_dir, labels_dir, train_file, val_file, test_file)
