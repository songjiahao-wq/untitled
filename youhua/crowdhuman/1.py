#更改数据集的分类
#将数据集txt的分类从1改为0

import os

label_dir_path = r'C:\Users\11097\Desktop\labels/'

label_dir = os.listdir(label_dir_path)
label_dir = [f for f in label_dir if f.endswith('.txt')]
for i in label_dir:
    label_path = r'C:\Users\11097\Desktop\labels/' + i
    ans = ''
    with open(label_path) as f:
        line = f.readline()
        while line:
            cls = int(line.split(' ')[0])
            if cls == 1:
                ans = ans + '0'+ line[1:]

            line = f.readline()
    print(ans)
    print('___________________________________-')
    with open(label_path, 'w') as outfile:
        outfile.write(ans)