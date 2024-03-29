import torch.nn as nn
import torch
from functools import reduce
from collections import OrderedDict, namedtuple
class SKAttention(nn.Module):

    def __init__(self, channel=512,kernels=[1,3,5,7],reduction=16,group=1,L=32):
        super().__init__()
        self.d=max(L,channel//reduction)
        self.convs=nn.ModuleList([])
        for k in kernels:
            self.convs.append(
                nn.Sequential(OrderedDict([
                    ('conv',nn.Conv2d(channel,channel,kernel_size=k,padding=k//2,groups=group)),
                    ('bn',nn.BatchNorm2d(channel)),
                    ('relu',nn.ReLU())
                ]))
            )
        self.fc=nn.Linear(channel * 21 ,self.d)
        self.fcs=nn.ModuleList([])
        for i in range(len(kernels)):
            self.fcs.append(nn.Linear(self.d,channel))
        self.softmax=nn.Softmax(dim=0)
        self.avg_pool1 =    nn.AdaptiveAvgPool2d(1)
        self.avg_pool2 =    nn.AdaptiveAvgPool2d(2)
        self.avg_pool4 =    nn.AdaptiveAvgPool2d(4)

    def forward(self, x):
        bs, c, _, _ = x.size()
        conv_outs=[]
        ### split
        for conv in self.convs:
            conv_outs.append(conv(x))
        feats=torch.stack(conv_outs,0)#k,bs,channel,h,w

        ### fuse
        U=sum(conv_outs) #bs,c,h,w

        y1 = self.avg_pool1(x).reshape((b, -1))
        y2 = self.avg_pool2(x).reshape((b, -1))
        y3 = self.avg_pool4(x).reshape((b, -1))
        S = torch.cat((y1, y2, y3), 1)
        ### reduction channel
        # S=U.mean(-1).mean(-1) #bs,c
        Z=self.fc(S) #bs,d

        ### calculate attention weight
        weights=[]
        for fc in self.fcs:
            weight=fc(Z)
            weights.append(weight.view(bs,c,1,1)) #bs,channel
        attention_weughts=torch.stack(weights,0)#k,bs,channel,1,1
        attention_weughts=self.softmax(attention_weughts)#k,bs,channel,1,1

        ### fuse
        V=(attention_weughts*feats).sum(0)
        return V
if __name__ =='__main__':
    x = torch.randn(1, 64, 32, 32)  # 用来生成随机数字的tensor 输出一个64*32*48的张量
    b, c, h, w = x.shape
    net1 = SKAttention(c)
    # net2 = SELayer(c, c)
    y1 = net1(x)
    print(y1.shape)
