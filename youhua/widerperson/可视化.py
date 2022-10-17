import os
import cv2

if __name__ == '__main__':
    path = 'D:\songjiahao\DATA\WiderPerson/train.txt'
    with open(path, 'r') as f:
        img_ids = [x for x in f.read().splitlines()]

    for img_id in img_ids:  # '000040'
        img_path = 'D:/songjiahao/DATA/WiderPerson/Images/' + img_id + '.jpg'
        img = cv2.imread(img_path)

        im_h = img.shape[0]
        im_w = img.shape[1]
        print(img_path)
        label_path = img_path.replace('Images','Annotations') + '.txt'
        print(label_path)
        with open(label_path) as file:
            line = file.readline()
            count = int(line.split('\n')[0])  # 里面行人个数
            line = file.readline()
            while line:
                cls = int(line.split(' ')[0])
                print(cls)
                # < class_label =1: pedestrians > 行人
                # < class_label =2: riders >      骑车的
                # < class_label =3: partially-visible persons > 遮挡的部分行人
                # < class_label =4: ignore regions > 一些假人，比如图画上的人
                # < class_label =5: crowd > 拥挤人群，直接大框覆盖了
                if cls == 1  or cls == 3:
                    xmin = float(line.split(' ')[1])
                    ymin = float(line.split(' ')[2])
                    xmax = float(line.split(' ')[3])
                    ymax = float(line.split(' ')[4].split('\n')[0])
                    img = cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
                line = file.readline()
        cv2.imshow('result', img)
        cv2.waitKey(0)
