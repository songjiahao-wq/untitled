import pickle
import numpy as np
import os
import matplotlib.pyplot as plt

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

data_folder = r"F:\sjh\分类\code\Train_Custom_Dataset\图像分类\3-【Pytorch】迁移学习训练自己的图像分类模型\data\cifar-10-batches-py/"
train_files = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
test_file = 'test_batch'

# Load training data
X_train = []
y_train = []
for train_file in train_files:
    data_dict = unpickle(os.path.join(data_folder, train_file))
    X_train.append(data_dict[b'data'])
    y_train += data_dict[b'labels']
X_train = np.concatenate(X_train).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1) / 255.0
y_train = np.array(y_train)

# Load test data
data_dict = unpickle(os.path.join(data_folder, test_file))
X_test = data_dict[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1) / 255.0
y_test = np.array(data_dict[b'labels'])
# Create class folders
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
for class_name in class_names:
    os.makedirs(os.path.join('cifar-10', class_name), exist_ok=True)

# Save training images
# for i in range(X_train.shape[0]):
#     file_name = os.path.join('cifar-10', class_names[y_train[i]], f'{i}.png')
#     plt.imsave(file_name, X_train[i])

# Save test images
for i in range(X_test.shape[0]):
    file_name = os.path.join('cifar-10', class_names[y_test[i]], f'{i}.png')
    plt.imsave(file_name, X_test[i])
