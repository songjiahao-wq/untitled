#更改数据集的分类
#将数据集txt的分类从1改为0

import os

label_dir = r'D:\songjiahao\DATA\Phone\phonecall_labels/'

label_dir = os.listdir(label_dir)
for i in label_dir:
    label_path = r'D:\songjiahao\DATA\Phone\phonecall_labels/' + i
    ans = ''
    with open(label_path) as f:
        line = f.readline()
        while line:
            cls = int(line.split(' ')[0])
            if cls == 0:
                ans = ans + '1'+ line[1:]

            line = f.readline()
    print(ans)
    print('___________________________________-')
    with open(label_path, 'w') as outfile:
        outfile.write(ans)