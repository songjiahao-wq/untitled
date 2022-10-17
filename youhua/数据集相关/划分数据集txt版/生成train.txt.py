import os
import random
import sys

if len(sys.argv) < 2:
    print("no directory specified, please input target directory")
    exit()

root_path = 'sys.argv[1]'

xmlfilepath = root_path + '/images'

txtsavepath = root_path + '/labels'

if not os.path.exists(root_path):
    print("cannot find such directory: " + root_path)
    exit()

if not os.path.exists(txtsavepath):
    os.makedirs(txtsavepath)

trainval_percent = 0.9  #train和val总共占比
train_percent = 0.8  #trainval里的train占比
total_xml = os.listdir(xmlfilepath)
num = len(total_xml)
list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)

print("train and val size:", tv)
print("train size:", tr)

ftrainval = open(root_path + '/trainval.txt', 'w')
ftest = open(root_path + '/test.txt', 'w')
ftrain = open(root_path + '/train.txt', 'w')
fval = open(root_path + '/val.txt', 'w')

for i in list:
    name = total_xml[i][:-4] + '\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)

ftrainval.close()
ftrain.close()
# fval.close()
# ftest.close()
