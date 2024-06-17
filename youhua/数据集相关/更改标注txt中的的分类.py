#更改数据集的分类
#将数据集txt的分类从1改为0

import os

label_source = r'E:\Download\Datasets\football\Football Analyzer.v1i.yolov8\train\labels/'

label_dir = os.listdir(label_source)
for i in label_dir:
    label_path = label_source + i
    ans = ''
    with open(label_path) as f:
        line = f.readline()
        while line:
            cls = int(line.split(' ')[0])
            if cls == 0:
                ans = ans + '80'+ line[1:]

            line = f.readline()
    print(ans)
    print('___________________________________-')
    with open(label_path, 'w') as outfile:
        outfile.write(ans)