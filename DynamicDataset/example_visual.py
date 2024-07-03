from n_mnist import NMNIST
from N_Caltech101 import NCaltech101
from cifar10_dvs import CIFAR10DVS
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch

#本包已下载好数据集。若 downloadable False说明不支持在不登陆的情况下使用代码直接下载，因此用户需要手动从网站上下载，此命令会打印出获取下载地址的网址
print('NMNIST downloadable', NMNIST.downloadable())
print('NMNIST, url, md5/n', NMNIST.resource_url_md5())

print('NCaltech101 downloadable', NCaltech101.downloadable())
print('NCaltech101, url, md5/n', NCaltech101.resource_url_md5())

print('CIFAR10-DVS downloadable', CIFAR10DVS.downloadable())
print('resource, url, md5/n', CIFAR10DVS.resource_url_md5())


batch_size = 10
#----------NMNIST-----------
#下载数据集 data_type可选frame默认event，设置 split_by='number' 表示从Event数量上进行划分，接近均匀地划分为 frames_num=20
nmnist_train = NMNIST(root='./NMNIST', train=True, data_type='frame', frames_number=20, split_by='number')
nmnist_test = NMNIST(root='./NMNIST', train=False, data_type='frame', frames_number=20, split_by='number')
#加载数据集
train_data_loader = DataLoader(dataset=nmnist_train, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0)
test_data_loader = DataLoader(dataset=nmnist_test, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0)
#输出示例图
dataiter = iter(train_data_loader)
images, labels = next(dataiter)
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i][1][1].numpy().squeeze())
    plt.title(labels[i].item())
plt.show()

nmnist_train = NMNIST(root='./NMNIST', train=True, data_type='event')
nmnist_test = NMNIST(root='./NMNIST', train=False, data_type='event')
#加载数据集
train_data_loader = DataLoader(dataset=nmnist_train, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0)
test_data_loader = DataLoader(dataset=nmnist_test, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0)
#输出示例图
dataiter = iter(train_data_loader)
images, labels = next(dataiter)
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i][1][1].type(torch.int32).numpy().squeeze())
    plt.title(labels[i].item())
plt.show()

#----------CIFAR10DVS----------
cifar10dvs = CIFAR10DVS(root='./CIFAR10DVS', data_type='frame', frames_number=20, split_by='number')
data_loader = DataLoader(dataset=cifar10dvs, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0)

dataiter = iter(data_loader)
images, labels = next(dataiter)
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i][1][1].numpy().squeeze())
    plt.title(labels[i].item())
plt.show()

#----------NCaltech101----------
nCaltech101 = NCaltech101(root='./Caltech101', data_type='frame', frames_number=20, split_by='number')
data_loader = DataLoader(dataset=nCaltech101, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0)

dataiter = iter(data_loader)
images, labels = next(dataiter)
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i][1][1].numpy().squeeze())
    plt.title(labels[i].item())
plt.show()