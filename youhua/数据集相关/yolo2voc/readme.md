voc
VOC**数据集由五个部分构成：JPEGImages，Annotations，ImageSets，SegmentationClass以及SegmentationObject.

* JPEGImages：存放的是训练与测试的所有图片。
* Annotations：里面存放的是每张图片打完标签所对应的XML文件。
* ImageSets：ImageSets文件夹下本次讨论的只有Main文件夹，此文件夹中存放的主要又有四个文本文件test.txt、train.txt、trainval.txt、val.txt, 其中分别存放的是测试集图片的文件名、训练集图片的文件名、训练验证集图片的文件名、验证集图片的文件名。
* SegmentationClass与SegmentationObject：存放的都是图片，且都是图像分割结果图，对目标检测任务来说没有用。class segmentation 标注出每一个像素的类别
* object segme**ntation 标注出每一个像素属于哪一个物体。