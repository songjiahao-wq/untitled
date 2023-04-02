from torch import nn
import torch
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# func = nn.Sigmoid()
# x = torch.arange(start=-10, end=10, step=0.01)
# y = func(x) * x
# plt.plot(x.numpy(), y.numpy())
# plt.title("silu")
# plt.savefig("silu.png")
# func = nn.Tanh()
# x = torch.arange(start=-10, end=10, step=0.01)
# y = func(x)
# plt.plot(x.numpy(), y.numpy())
# plt.title("tanh")
# plt.savefig("tanh.png")
func = nn.Sigmoid()
x = torch.arange(start=-10, end=10, step=0.01)
y = func(x)
plt.plot(x.numpy(), y.numpy())
plt.title("sigmoid")
plt.savefig("sigmoid.png")
