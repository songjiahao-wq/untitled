import os
txtpath = r'D:\xian_yu\4.27\yolov4-pytorch-master\2007_val.txt'

file_data = ''
with open(txtpath, 'r') as f:
    lines = f.readlines()
    for line in lines:

        newline = line.replace(r"F:\PycharmProjects\pythonProject\bili\yolov4-pytorch-master", r"D:/xian_yu/4.27/yolov4-pytorch-master")
        newline = newline.replace('\\', '/')
        file_data +=newline
with open(txtpath, 'w') as f:
    f.write(file_data)
