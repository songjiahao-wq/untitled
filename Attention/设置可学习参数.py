import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

# 网络结构
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 设置可学习权值
        self.w1 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w2 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)

        # 初始化权值
        self.w1.data.fill_(0.5)
        self.w2.data.fill_(0.5)

        # 两个卷积层
        self.conv1 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1, 1)

    def forward(self, x):
        x = self.w1 * self.conv1(x) + self.w2 * self.conv2(x)
        return F.relu(x)

epoch = 100
learning_rate = 1e-3 #学习率

myNet = Net() #定义网络
optimizer = Adam(myNet.parameters(), lr=learning_rate) #定义优化器
loss_fun = nn.MSELoss() #定义loss

x = torch.randn([10, 32, 64, 64]) #定义输入
y = torch.randn([10, 32, 64, 64]) #定义输出

# 开始训练
for i in range(epoch):
    # 前向计算和损失函数
    output = myNet(x)
    loss = loss_fun(output, y) #计算loss，参数（输入，输出）

    # 清空梯度信息
    optimizer.zero_grad()

    # 反向传播和参数更新
    loss.backward()  # 反向传播
    optimizer.step()

    print(f'权值1：{myNet.w1.item()}')
    print(f'权值2：{myNet.w2.item()}')
