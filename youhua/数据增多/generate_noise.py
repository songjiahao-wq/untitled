# -*- coding: utf-8 -*-
import glob
import numpy as np
import torch
from torch.utils import data
from PIL import Image
from torchvision import transforms as T
from torch.utils.data import DataLoader
import random
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
num_data = 100
class Train_Data(data.Dataset):
    def __init__(self, data_root):

        self.transform = T.ToTensor()
        self.transform1 = T.ToPILImage()
        self.data_root = './data/imgs'
        self.density = 0.2

    def __getitem__(self, index):

        in_files = glob.glob(r'F:\sjh\DATA2\NEU\images\val15/*.jpg')
        for i, filename in enumerate(in_files):

            img = Image.open(filename)
            
            # this is to add gaussian noise
            label = self.transform(img)
            noise = add_noise(label, 15)  # noise level: 15 25 50
            noise = noise.permute(1, 2, 0)
            noise = np.array(noise)
            out_filename = filename
            # out_filename = './data1/noise/' + out_filename
            mpimg.imsave(out_filename, noise)
            
            # this is to add the salt noise
            # img = np.array(img)  # 图片转numpy
            # h, w, c = img.shape
            # Nd = self.density
            # Sd = 1 - Nd
            # mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[Nd / 2.0, Nd / 2.0, Sd])  # 生成一个通道的mask
            # mask = np.repeat(mask, c, axis=2)  # 在通道的维度复制，生成彩色的mask
            # img[mask == 0] = 0  # 椒
            # img[mask == 1] = 255  # 盐
            # img = Image.fromarray(img.astype('uint8')).convert('RGB')  # numpy转图片
            # out_filename = filename
            # # out_filename = './data1/noise/' + out_filename
            # img.save(out_filename)
        return img

    def __len__(self):
        return num_data

def add_noise(input_img, noise_sigma):
    noise_sigma = noise_sigma / 255
    noise_img = torch.clamp(input_img+noise_sigma*torch.randn_like(input_img), 0.0, 1.0)

    return noise_img

if __name__ == '__main__':
    train_data = Train_Data(data_root='./data/cbsd68')

    train_loader = DataLoader(train_data, 1)

    for i, (data, label) in enumerate(train_data):
        print(i)
        if i == 0:
            print(data)
            print(label)
            print(data.size())
            print(label.size())
            break





