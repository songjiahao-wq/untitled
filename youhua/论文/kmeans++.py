
import glob
import xml.etree.ElementTree as ET

import numpy as np

def cas_iou(box ,cluster):
    x = np.minimum(cluster[: ,0] ,box[0])
    y = np.minimum(cluster[: ,1] ,box[1])

    intersection = x * y
    area1 = box[0] * box[1]

    area2 = cluster[: ,0] * cluster[: ,1]
    iou = intersection / (area1 + area2 -intersection)

    return iou


def avg_iou(box, cluster):
    return np.mean([np.max(cas_iou(box[i], cluster)) for i in range(box.shape[0])])


def bboxesOverRation(bboxesA, bboxesB):
    bboxesA = np.array(bboxesA.astype('float'))
    bboxesB = np.array(bboxesB.astype('float'))
    M = bboxesA.shape[0]
    N = bboxesB.shape[0]

    areasA = bboxesA[:, 2] * bboxesA[:, 3]
    areasB = bboxesB[:, 2] * bboxesB[:, 3]

    xA = bboxesA[:, 0] + bboxesA[:, 2]
    yA = bboxesA[:, 1] + bboxesA[:, 3]
    xyA = np.stack([xA, yA]).transpose()
    xyxyA = np.concatenate((bboxesA[:, :2], xyA), axis=1)

    xB = bboxesB[:, 0] + bboxesB[:, 2]
    yB = bboxesB[:, 1] + bboxesB[:, 3]
    xyB = np.stack([xB, yB]).transpose()
    xyxyB = np.concatenate((bboxesB[:, :2], xyB), axis=1)

    iouRatio = np.zeros((M, N))
    for i in range(M):
        for j in range(N):
            x1 = max(xyxyA[i, 0], xyxyB[j, 0]);
            x2 = min(xyxyA[i, 2], xyxyB[j, 2]);
            y1 = max(xyxyA[i, 1], xyxyB[j, 1]);
            y2 = min(xyxyA[i, 3], xyxyB[j, 3]);
            Intersection = max(0, (x2 - x1)) * max(0, (y2 - y1));
            Union = areasA[i] + areasB[j] - Intersection;
            iouRatio[i, j] = Intersection / Union;
    return iouRatio


def load_data(path):
    data = []
    # 对于每一个xml都寻找box
    for xml_file in glob.glob('{}/*xml'.format(path)):
        tree = ET.parse(xml_file)
        height = int(tree.findtext('./size/height'))
        width = int(tree.findtext('./size/width'))
        if height <= 0 or width <= 0:
            continue

        # 对于每一个目标都获得它的宽高
        for obj in tree.iter('object'):
            xmin = int(float(obj.findtext('bndbox/xmin'))) / width
            ymin = int(float(obj.findtext('bndbox/ymin'))) / height
            xmax = int(float(obj.findtext('bndbox/xmax'))) / width
            ymax = int(float(obj.findtext('bndbox/ymax'))) / height

            xmin = np.float64(xmin)
            ymin = np.float64(ymin)
            xmax = np.float64(xmax)
            ymax = np.float64(ymax)
            # 得到宽高
            x = xmin + 0.5 * (xmax - xmin)
            y = ymin + 0.5 * (ymax - ymin)
            data.append([x, y, xmax - xmin, ymax - ymin])
    return np.array(data)


def estimateAnchorBoxes(trainingData, numAnchors=9, SIZE=416):
    numsObver = trainingData.shape[0]
    xyArray = np.zeros((numsObver, 2))
    trainingData[:, 0:2] = xyArray
    assert (numsObver >= numAnchors)

    # kmeans++
    # init 
    centroids = []  # 初始化中心，kmeans++
    centroid_index = np.random.choice(numsObver, 1)
    centroids.append(trainingData[centroid_index])
    while len(centroids) < numAnchors:
        minDistList = []
        for box in trainingData:
            box = box.reshape((-1, 4))
            minDist = 1
            for centroid in centroids:
                centroid = centroid.reshape((-1, 4))
                ratio = (1 - bboxesOverRation(box, centroid)).item()
                if ratio < minDist:
                    minDist = ratio
            minDistList.append(minDist)

        sumDist = np.sum(minDistList)
        prob = minDistList / sumDist
        idx = np.random.choice(numsObver, 1, replace=True, p=prob)
        centroids.append(trainingData[idx])

    # kmeans 迭代聚类
    maxIterTimes = 100
    iter_times = 0
    while True:
        minDistList = []
        minDistList_ind = []
        for box in trainingData:
            box = box.reshape((-1, 4))
            minDist = 1
            box_belong = 0
            for i, centroid in enumerate(centroids):
                centroid = centroid.reshape((-1, 4))
                ratio = (1 - bboxesOverRation(box, centroid)).item()
                if ratio < minDist:
                    minDist = ratio
                    box_belong = i
            minDistList.append(minDist)
            minDistList_ind.append(box_belong)
        centroids_avg = []
        for _ in range(numAnchors):
            centroids_avg.append([])
        for i, anchor_id in enumerate(minDistList_ind):
            centroids_avg[anchor_id].append(trainingData[i])
        err = 0
        for i in range(numAnchors):
            if len(centroids_avg[i]):
                temp = np.mean(centroids_avg[i], axis=0)
                err += np.sqrt(np.sum(np.power(temp - centroids[i], 2)))
                centroids[i] = np.mean(centroids_avg[i], axis=0)
        iter_times += 1
        if iter_times > maxIterTimes or err == 0:
            break
    anchorBoxes = np.array([x[2:] for x in centroids])
    meanIoU = 1 - np.mean(minDistList)
    anchorBoxes = anchorBoxes[np.argsort(anchorBoxes[:, 0])]
    print('acc:{:.2f}%'.format(avg_iou(trainingData[:, 2:], anchorBoxes) * 100))
    anchorBoxes = anchorBoxes * SIZE
    return anchorBoxes, meanIoU


def calculate_anchors(dataset_anno_path=r'/home/slave110/code/VOCdevkit/VOC2007/Annotations',
                      anchorsPath='./yolo_anchors.txt', anchors_num=9, SIZE=1000):
    data = load_data(dataset_anno_path)
    anchors, _ = estimateAnchorBoxes(data, numAnchors=anchors_num, SIZE=SIZE)
    print(anchors)
    f = open(anchorsPath, 'w')
    row = np.shape(anchors)[0]
    for i in range(row):
        if i == 0:
            x_y = "%d,%d" % (anchors[i][0], anchors[i][1])
        else:
            x_y = ", %d,%d" % (anchors[i][0], anchors[i][1])
        f.write(x_y)
    f.close()


if __name__ == "__main__":
    # 数据集xml注释路径
    dataset_anno_path = r'D:\songjiahao\DATA\crowhuman-voc\VOC2007\Annotations'
    # 生成的anchors的txt文件保存路径
    anchorsPath = 'D:\songjiahao\DATA\crowhuman-voc\VOC2007/anchors.txt'
    # 生成的anchors数量
    anchors_num = 9
    # 输入的图片尺寸
    SIZE = 1000
    calculate_anchors(dataset_anno_path, anchorsPath, anchors_num, SIZE)   
