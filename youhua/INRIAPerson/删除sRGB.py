import os
from tqdm import tqdm
import cv2
from skimage import io
#import os
path = r"D:\my_job\DATA\INRIAPerson\inriaPerson\test/" #path后面记得加 /
#西瓜6的代码
fileList = os.listdir(path)
fileList = [i  for i in fileList if i.endswith('.png')]
for i in tqdm(fileList):
    image = io.imread(path+i)  # image = io.imread(os.path.join(path, i))
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
    cv2.imencode('.png',image)[1].tofile(path+i)
