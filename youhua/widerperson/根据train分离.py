#第一天学习
#第一天学习
# 根据train。txt和test。txt划分数据集
import os
import shutil

#原始路径
traintxt=r'D:\songjiahao\DATA\数据集\WiderPerson\train.txt'
testtxt=r'D:\songjiahao\DATA\数据集\WiderPerson\val.txt'
srcima=r'D:\songjiahao\DATA\数据集\widerhuman\images\train/'
srctxt=r'D:\songjiahao\DATA\数据集\widerhuman\labels\train/'
#训练集路径
train_image_path = r'D:\songjiahao\DATA\数据集\WiderPerson\widerPerson划分\images\train/'
train_label_path = r'D:\songjiahao\DATA\数据集\WiderPerson\widerPerson划分\labels\train/'
# 验证集路径
val_image_path = r'D:\songjiahao\DATA\数据集\WiderPerson\widerPerson划分\images\val/'
val_label_path = r'D:\songjiahao\DATA\数据集\WiderPerson\widerPerson划分\labels\val/'
# 测试集路径
test_image_path = r'D:\songjiahao\DATA\数据集\WiderPerson\widerPerson划分\images\test/'
test_label_path = r'D:\songjiahao\DATA\数据集\WiderPerson\widerPerson划分\labels\test/'
# 数据集划分比例，训练集75%，验证集15%，测试集15%
train_percent = 0.7
val_percent = 0.1
test_percent = 0.1
# 检查文件夹是否存在
def mkdir():
    if not os.path.exists(train_image_path):

        os.makedirs(train_image_path)
    if not os.path.exists(train_label_path):
        os.makedirs(train_label_path)

    if not os.path.exists(val_image_path):
        os.makedirs(val_image_path)
    if not os.path.exists(val_label_path):
        os.makedirs(val_label_path)

    if not os.path.exists(test_image_path):
        os.makedirs(test_image_path)
    if not os.path.exists(test_label_path):
        os.makedirs(test_label_path)

def main():
    mkdir()
    f = open(traintxt, 'r')
    trainlist = f.readlines()
    f = open(testtxt, 'r')
    testlist = f.readlines()
    f.close()
    print("训练集数目：{}，测试集数目：{}".format(len(trainlist),len(testlist)))
    for i in range(len(trainlist)):

        name = trainlist[i].strip()
        # srcImage = traintxt.replace('train.txt','')  + name + '.jpg'
        # srcLabel = traintxt.replace('train.txt','') + name + '.txt'
        srcImage = srcima + name + '.jpg'
        srcLabel = srctxt + name +'.txt'
        print(srcImage)
        dst_train_Image = train_image_path + name + '.jpg'
        dst_train_Label = train_label_path + name + '.txt'

        shutil.copyfile(srcImage, dst_train_Image)
        print(srcImage)
        # shutil.copyfile(srcLabel, dst_train_Label)
    for i in range(len(testlist)):
        name = testlist[i].strip()

        srcImage = srcima + name + '.jpg'
        srcLabel = srctxt + name + '.txt'

        dst_test_Image = test_image_path + name + '.jpg'
        dst_test_Label = test_label_path + name + '.txt'

        # shutil.copyfile(srcImage, dst_test_Image)
        # shutil.copyfile(srcLabel, dst_test_Label)
        # print(i+1)



if __name__ == '__main__':
    main()