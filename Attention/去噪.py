import torch
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import torchvision
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),  # (batch, 64, H/2, W/2)
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # (batch, 128, H/4, W/4)
            nn.ReLU(),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # (batch, 64, H/2, W/2)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),  # (batch, 3, H, W)
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

if __name__ =="__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Hyperparameters
    num_epochs = 100
    batch_size = 64
    learning_rate = 0.001
    noise_factor = 0.5

    # Initialize the model, loss, and optimizer
    model = ConvAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 定义添加高斯噪声的函数
    def add_gaussian_noise(x):
        return x + 0.1 * torch.randn_like(x)


    # 定义截断函数
    def clamp(x):
        return torch.clamp(x, 0, 1)


    # 转换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(add_gaussian_noise),
        transforms.Lambda(clamp),
    ])


    # # 加载数据集
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    val_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    #
    # # Load dataset
    # # train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    # # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    #

    #
    # # Training loop
    # for epoch in range(num_epochs):
    #     for i, (noisy_imgs, _) in enumerate(train_loader):
    #         # Forward pass
    #         noisy_imgs = noisy_imgs.to(device)
    #         outputs = model(noisy_imgs)
    #         # print(outputs.device,noisy_imgs.device)
    #         loss = criterion(outputs, noisy_imgs)
    #
    #         # Backward pass
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #
    #         if (i + 1) % 100 == 0:
    #             print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item()}')
    #
    # print('Training finished.')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    num_epochs = 100
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # 初始化最佳验证损失为无穷大
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        total_loss = 0
        # Training
        model.train()
        train_loss = 0
        for inputs, _ in train_loader:
            inputs = inputs.to(device)
            noisy_inputs = inputs + noise_factor * torch.randn_like(inputs)
            noisy_inputs = torch.clamp(noisy_inputs, 0., 1.)

            outputs = model(noisy_inputs)
            loss = criterion(outputs, inputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, _ in val_loader:
                inputs = inputs.to(device)
                noisy_inputs = inputs + noise_factor * torch.randn_like(inputs)
                noisy_inputs = torch.clamp(noisy_inputs, 0., 1.)

                outputs = model(noisy_inputs)
                loss = criterion(outputs, inputs)

                val_loss += loss.item()

            # 计算平均验证损失
        val_loss = val_loss / len(val_loader)

        # 如果当前验证损失小于之前的最佳损失，则更新最佳损失并保存模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_denoising_autoencoder.pth')
            print(f"Epoch: {epoch + 1}, Validation loss improved. Saving the model.")
        else:
            print(f"Epoch: {epoch + 1}, Validation loss did not improve.")

        print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {val_loss:.4f}")

"""
要将自编码器或GAN作为输入层集成到YOLOv5中，可以按照以下步骤操作：

首先，训练一个自编码器或GAN模型。在训练过程中，您将学习一个能够有效去噪的模型。确保保存训练好的模型权重以供稍后使用。

修改YOLOv5的网络配置。在YOLOv5的配置文件中，添加一个新的自定义层，该层将加载训练好的自编码器或GAN模型并将其作为输入层。这可以通过在backbone部分的开头插入一个自定义层实现。例如，对于自编码器，您可以这样做：
backbone:
  # [from, number, module, args]
  [[-1, 1, CustomDenoisingLayer, [your_encoder_path]],  # Add custom denoising layer as input layer
   [-1, 1, Conv, [64, 6, 2, 2]],  # 0-P1/2
   ...
其中CustomDenoisingLayer是您将在代码中实现的自定义层，而your_encoder_path是训练好的自编码器或GAN模型的权重路径。
"""