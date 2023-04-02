import torch.nn as nn
class net4(nn.Module):
    def __init__(self):
        super(net4, self).__init__()
        self.linears = nn.ModuleList([nn.Conv2d(64,64,3,1),nn.Conv2d(64,64,5,3)])
    def forward(self, x):
        x = self.linears[0](x)
        # x = self.linears[1](x)
        # x = self.linears[1](x)
        return x
net = net4()
print(net)
# net4(
#   (linears): ModuleList(
#     (0): Linear(in_features=5, out_features=10, bias=True)
#     (1): Linear(in_features=10, out_features=10, bias=True)
#   )
# )
for name, param in net.named_parameters():

    print(name, param.size())
# linears.0.weight torch.Size([10, 5])
# linears.0.bias torch.Size([10])
# linears.1.weight torch.Size([10, 10])
# linears.1.bias torch.Size([10])