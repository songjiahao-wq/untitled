#更改数据集的分类
#将数据集txt的分类从1改为0

import os

label_dir = r'F:\sjh\DATA2\xianyu\smoke_safe hat\labels\val/'

label_dir = os.listdir(label_dir)
for i in label_dir:
    label_path = r'F:\sjh\DATA2\xianyu\smoke_safe hat\labels\val/' + i
    ans = ''
    with open(label_path) as f:
        line = f.readline()
        while line:
            cls = int(line.split(' ')[0])
            if cls == 0:
                ans = ans + '2'+ line[1:]

            line = f.readline()
    print(ans)
    print('___________________________________-')
    with open(label_path, 'w') as outfile:
        outfile.write(ans)