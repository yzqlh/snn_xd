import matplotlib.pyplot as plt
import numpy as np
from StaticDatasets import MNIST, FashionMNIST, CIFAR10, CIFAR100
from cityscapes_ import CityscapesDataset
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms


#MNIST
#加载数据集
train_transforms = transforms.ToTensor()
test_transforms = transforms.ToTensor()
mnist_instance = MNIST(64, train_transforms, test_transforms)
train_loader_test, test_loader_test = mnist_instance.train_loader, mnist_instance.test_loader

#可视化
dataiter = iter(train_loader_test)
images, labels = next(dataiter)
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i].numpy().squeeze(), cmap='gray')
    plt.title(labels[i].item())
plt.show()

#FashionMNIST
#加载数据集
fmnist_instance = FashionMNIST(64, train_transforms, test_transforms)
train_loader_test, test_loader_test = fmnist_instance.train_loader, fmnist_instance.test_loader

#可视化
dataiter = iter(train_loader_test)
images, labels = next(dataiter)
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i].numpy().squeeze())  # cmap='gray'
    plt.title(labels[i].item())
plt.show()

#CIFAR10
#加载数据集
CIFAR10_instance = CIFAR10(64, train_transforms, test_transforms)
train_loader_test, test_loader_test = CIFAR10_instance.train_loader, CIFAR10_instance.test_loader

#可视化
dataiter = iter(train_loader_test)
images, labels = next(dataiter)
# 展示图像和标签
#ncols 生成图片数量
fig, axes = plt.subplots(figsize=(10, 5), ncols=4)
for i in range(4):
    ax = axes[i]
    ax.imshow(np.transpose(images[i], (1, 2, 0)))
    ax.set_title(labels[i].item())# 设置图片的标签
    ax.axis('off')
plt.show()

#CIFAR100
#加载数据集
CIFAR100_instance = CIFAR100(64, train_transforms, test_transforms)
train_loader_test, test_loader_test = CIFAR100_instance.train_loader, CIFAR100_instance.test_loader

#可视化
dataiter = iter(train_loader_test)
images, labels = next(dataiter)
# 展示图像和标签
#ncols 生成图片数量
fig, axes = plt.subplots(figsize=(10, 5), ncols=4)
for i in range(4):
    ax = axes[i]
    ax.imshow(np.transpose(images[i], (1, 2, 0)))
    ax.set_title(labels[i].item())# 设置图片的标签
    ax.axis('off')
plt.show()



#Cityscape
# 设置数据集路径
x_train_dir = r"D:\Dataset\cityscapesdata\leftImg8bit\cityscapes_train"
y_train_dir = r"D:\Dataset\cityscapesdata\gtFine\cityscapes_19classes_train"

x_valid_dir = r"D:\Dataset\cityscapesdata\leftImg8bit\cityscapes_val"
y_valid_dir = r"D:\Dataset\cityscapesdata\gtFine\cityscapes_19classes_val"

train_dataset = CityscapesDataset(
    x_train_dir,
    y_train_dir,
)
val_dataset = CityscapesDataset(
    x_valid_dir,
    y_valid_dir,
)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True)

for index, (img, label) in enumerate(train_loader):
    print(img.shape)
    print(label.shape)

    plt.figure(figsize=(10, 10))
    plt.subplot(221)
    plt.imshow((img[0, :, :, :].moveaxis(0, 2)))
    plt.subplot(222)
    plt.imshow(label[0, :, :])

    plt.subplot(223)
    plt.imshow((img[6, :, :, :].moveaxis(0, 2)))
    plt.subplot(224)
    plt.imshow(label[6, :, :])
    plt.show()
    if index == 0:
        break
