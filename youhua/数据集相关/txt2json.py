import mmcv
import os.path as osp
import os
"""
    --pest:
        --images:
        --imagesets:
            --test.txt
            --train.txt
            --val.txt
        --labels:
"""

def convert_txt_to_json(ann_file, out_file, image_prefix):
    """
    Args:
        ann_file: 标注文件 如：train.txt 里面保存了所有的训练图片（标签）文件名 如：20210819B000001
        out_file: 输出json格式文件名 这里根据自己的数据集位置做调整 如：../../../datasets/pest/jsons/val.json
        img_prefix: 数据集图片存放地址 这里根据自己的数据集位置做调整 如：../../../datasets/pest/images
    Returns:
    """
    json_path = os.path.abspath(os.path.join(out_file, ".."))  # json目录
    if os.path.exists(json_path) is False:  # 没有就新建
        os.makedirs(json_path)
    image_list = mmcv.list_from_file(ann_file)  # 读取txt文件中所有数据 按行读取

    image_id = 1  # 图片初始id
    annotation_id = 1  # 标注文件初始id

    coco_output = {
        "images": [],  # 存放所有图片信息
        "categories": [],  # 存放数据集类别信息
        "annotations": []  # 存放所有标注文件信息
    }

    categories = [
        {'id': 0, 'name': 'powdery_mildew'},
        {'id': 1, 'name': 'leaf_miner'},
        {'id': 2, 'name': 'anthracnose'},
    ]

    coco_output['categories'] = categories

    # mmcv.track_iter_progress：进度条
    for idx, img_name in enumerate(mmcv.track_iter_progress(image_list)):
        filename = f'{image_prefix}/{img_name}.jpg'  # 当前image图片地址
        image = mmcv.imread(filename)  # 读取当前图片
        height, width = image.shape[:2]  # 取得当前图片高、宽信息
        # 当前图片信息
        image_dict = {
            "file_name": f'{img_name}.jpg',
            "height": height,
            "width": width,
            "id": image_id,
        }
        # 将当前图片信息加入到coco_output中
        coco_output['images'].append(image_dict)

        # 读取当前图片对应的标注信息 并处理
        label_prefix = image_prefix.replace('images', 'labels')  # 获得当前图片对应的label文件地址
        lines = mmcv.list_from_file(osp.join(label_prefix, f'{img_name}.txt'))  # 读取当前label文件(txt格式）
        content = [line.strip().split(' ') for line in lines]
        category_ids = [x[0] for x in content]  # 获取当前label中所有object的类别
        bboxes = [[float(info) for info in x[1:]] for x in content]  # 获取当前label中所有object的bbox位置信息

        # 遍历当前图片的所有object
        for category_id, bbox in zip(category_ids, bboxes):
            # 更新bbox  xywh(normalization and float) -> xywh(no normalization and int)
            bbox[0] = int(bbox[0] * width)
            bbox[1] = int(bbox[1] * height)
            bbox[2] = int(bbox[2] * width)
            bbox[3] = int(bbox[3] * height)
            ann_dict = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": int(category_id),
                "bbox": bbox,
                "area": float(bbox[2]) * float(bbox[3]),
                "iscrowd": 0,
            }
            coco_output["annotations"].append(ann_dict)  # 更新当前图片的object label信息
            annotation_id += 1  # label id+1
        image_id += 1  # 图片id+1
    mmcv.dump(coco_output, out_file)  # 保存train.json or val.json or test.json to out_file

if __name__ == '__main__':
    convert_txt_to_json('../../../datasets/pest/imagesets/train.txt',
                        '../../../datasets/pest/jsons/train.json',
                        '../../../datasets/pest/images')
    # convert_txt_to_json('../../../datasets/pest/imagesets/val.txt',
    #                     '../../../datasets/pest/jsons/val.json',
    #                     '../../../datasets/pest/images')
    # convert_txt_to_json('../../../datasets/pest/imagesets/test.txt',
    #                     '../../../datasets/pest/jsons/test.json',
    #                     '../../../datasets/pest/images')
