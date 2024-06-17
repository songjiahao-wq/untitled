# -*- coding: utf-8 -*-
# @Time    : 2023/11/23 11:52
# @Author  : sjh
# @Site    : 
# @File    : softnms.py
# @Comment :
from torch import Tensor
import torch
import torchvision


def box_area(boxes: Tensor) -> Tensor:
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    area1 = box_area(boxes1)  # 每个框的面积 (N,)
    area2 = box_area(boxes2)  # (M,)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2] # N中一个和M个比较； 所以由N，M 个
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]  # 删除面积小于0 不相交的  clamp 钳；夹钳；
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]  # 切片的用法 相乘维度减1
    iou = inter / (area1[:, None] + area2 - inter)
    return iou  # NxM， boxes1中每个框和boxes2中每个框的IoU值；


def soft_nms(boxes: Tensor, scores: Tensor, soft_threshold=0.01, iou_threshold=0.7, weight_method=1, sigma=0.5):
    """
    :param boxes: [N, 4]， 此处传进来的框，是经过筛选（选取的得分TopK）之后的
    :param scores: [N]
    :param iou_threshold: 0.7
    :param soft_threshold soft nms 过滤掉得分太低的框 （手动设置）
    :param weight_method 权重方法 1. 线性 2. 高斯
    :return:
    """
    keep = []
    idxs = scores.argsort()
    while idxs.numel() > 0:  # 循环直到null； numel()： 数组元素个数
        # 由于scores得分会改变，所以每次都要重新排序，获取得分最大值
        idxs = scores.argsort()  # 评分排序
        if idxs.size(0) == 1:  # 就剩余一个框了；
            keep.append(idxs[-1])
            break
        keep_len = len(keep)
        # 例如idxs一共4个值，进行一轮之后只看前3个了，再一轮之后只看前2个了....
        # 后面的那些并不像以前一样直接删掉，因为可能每次乘了一些值之后又往前提了
        max_score_index = idxs[-(keep_len + 1)]
        max_score_box = boxes[max_score_index][None, :]  # [1, 4]
        idxs = idxs[:-(keep_len + 1)]
        other_boxes = boxes[idxs]  # [?, 4]
        keep.append(max_score_index)  # 位置不能边
        ious = box_iou(max_score_box, other_boxes)  # 一个框和其余框比较 1XM
        # Soft NMS 处理， 和 得分最大框 IOU大于阈值的框， 进行得分抑制
        if weight_method == 1:  # 线性抑制  # 整个过程 只修改分数
            ge_threshod_idxs = idxs[ious[0] >= iou_threshold]
            scores[ge_threshod_idxs] *= (1. - ious[0][ious[0] >= iou_threshold])  # 小于IoU阈值的不变
            # idxs = idxs[scores[idxs] >= soft_threshold]  # 小于soft_threshold删除， 经过抑制后 阈值会越来越小；
        elif weight_method == 2:  # 高斯抑制， 不管大不大于阈值，都计算权重
            scores[idxs] *= torch.exp(-(ious[0] * ious[0]) / sigma)  # 权重(0, 1]
            # idxs = idxs[scores[idxs] >= soft_threshold]
        print(idxs)
    keep = idxs.new(keep)  # Tensor
    keep = keep[scores[keep] > soft_threshold]  # 最后处理阈值
    boxes = boxes[keep]  # 保留下来的框
    scores = scores[keep]  # soft nms抑制后得分
    return boxes, scores


box = torch.tensor([[2, 3.1, 7, 5], [3, 4, 8, 4.8], [4, 4, 5.6, 7], [0.1, 0, 8, 1]])
score = torch.tensor([0.5, 0.3, 0.2, 0.4])

output = soft_nms(boxes=box, scores=score, iou_threshold=0.3)
print('IOU of bboxes:')
print(output)