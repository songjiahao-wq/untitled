import os
import shutil

# 路径配置
data_dir = r'D:\xian_yu\xianyun_lunwen\吸烟和打手机识别\大论文抽打数据集\final_data_small'  # 修改为您数据集的路径
vocdevkit_dir = r'D:\xian_yu\xianyun_lunwen\吸烟和打手机识别\大论文抽打数据集\final_data_small/VOCdevkit'  # 修改为VOCdevkit的路径

# 创建VOC格式的目录结构
voc_dirs = {
    'Annotations': os.path.join(vocdevkit_dir, 'VOC2007', 'Annotations'),
    'ImageSets': {
        'Main': os.path.join(vocdevkit_dir, 'VOC2007', 'ImageSets', 'Main')
    },
    'JPEGImages': os.path.join(vocdevkit_dir, 'VOC2007', 'JPEGImages')
}

for key, path in voc_dirs.items():
    if key == 'ImageSets':
        for subkey, subpath in path.items():
            os.makedirs(subpath, exist_ok=True)
    else:
        os.makedirs(path, exist_ok=True)


# 函数定义：写入ImageSets文件
def write_imagesets(imagesets_path, folder, file_list):
    with open(os.path.join(imagesets_path, folder + '.txt'), 'w') as file:
        for item in file_list:
            file.write(item + '\n')


# 函数定义：复制图像和注释到VOC结构
def copy_files(src_dir, dst_dir, files, ext):
    for fname in files:
        src_file = os.path.join(src_dir, fname + ext)
        dst_file = os.path.join(dst_dir, fname + ext)
        shutil.copy2(src_file, dst_file)


# 遍历原始数据集目录，处理每个子集（train, val, test）
for subset in ['train', 'valid', 'test']:
    images_path = os.path.join(data_dir, subset, 'images')
    annots_path = os.path.join(data_dir, subset, 'Annotations')

    # 获取图像文件名（无扩展名）
    image_files = [os.path.splitext(f)[0] for f in os.listdir(images_path) if f.endswith('.jpg')]

    # 复制图像和注释到VOC目录结构
    copy_files(images_path, voc_dirs['JPEGImages'], image_files, '.jpg')
    copy_files(annots_path, voc_dirs['Annotations'], image_files, '.xml')

    # 写入ImageSets文件
    if subset == 'train':
        # trainval是train和val的组合
        trainval_files = image_files
    elif subset == 'valid':
        # val和trainval更新
        val_files = image_files
        trainval_files.extend(image_files)
    elif subset == 'test':
        # test文件
        test_files = image_files

    # 保存ImageSets文件
    write_imagesets(voc_dirs['ImageSets']['Main'], subset, image_files)

# 最后写入trainval和val的ImageSets文件
write_imagesets(voc_dirs['ImageSets']['Main'], 'trainval', trainval_files)
write_imagesets(voc_dirs['ImageSets']['Main'], 'val', val_files)
write_imagesets(voc_dirs['ImageSets']['Main'], 'test', test_files)

print("VOC dataset has been set up at:", vocdevkit_dir)
