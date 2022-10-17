import os
from glob import glob
from pathlib import Path
from sklearn.model_selection import train_test_split
# 根据已有的txt生成train.txt文件
# 1.标签路径

shuju_path = r"D:\songjiahao\DATA\crowhuman/" +'ImageSets/Main/' #1 保存路径

# 3.split files for txt
txtsavepath = shuju_path
if not os.path.exists(txtsavepath):
    os.makedirs(txtsavepath)
for file_name in ['train','val','test']:
    total_files_path = os.path.join(r'D:\songjiahao\DATA\crowhuman\labels' , str(file_name))#2

    ftrain = open(txtsavepath + '/'+str(file_name)+'.txt', 'w')#3


    # total_files_path = os.path.join(shuju_path , "Annotations/Val")#2

    total_files = os.listdir(total_files_path)
    # print(total_files)
    num=len(total_files)
    list=range(num)
    print(list)
    # train
    for i in list:
        name = total_files[i] +'\n'
        txt_path = os.path.join(total_files_path,name)
        txt_path=txt_path.replace('\\','/')
        # txt_path = Path(txt_path).as_posix()
        # print(txt_path)
        ftrain.write(txt_path)


    ftrain.close()


