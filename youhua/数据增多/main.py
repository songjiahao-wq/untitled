#第一天学习
#第一天学习
from .数据集扩增 import random_perspective
img, labels = random_perspective(img, labels,
                                 degrees=hyp['degrees'],
                                 translate=hyp['translate'],
                                 scale=hyp['scale'],
                                 shear=hyp['shear'],
                                 perspective=hyp['perspective'])