import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from spikingjelly.activation_based import functional,neuron, monitor, base, layer, STBP
import torch.nn as nn
from spikingjelly.Dataset_encoding import MNIST_encoder, Cifar10_encoder, Cifar100_encoder, FashionMNIST_encoder


def train_global(net,device, batch_size, time, dataset_name:str = 'MNIST', encoder:str = 'poisson'):
    num = 0
    net = net
    functional.set_step_mode(net, 'm')
    loader = get_loader('train', dataset_name, encoder, batch_size, time)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    print(f'-----------------{num}-----------------')
    for img, label in loader:
        optimizer.zero_grad()
        img = img.to(device)
        label = label.to(device)
        label_onehot = F.one_hot(label, 10).float()

        img = img.permute(1, 0, 2, 3, 4)
        out_fr1 = 0.
        out_fr1 = net(img)
        out_fr1 = out_fr1.sum(dim=0) / time

        loss = F.mse_loss(out_fr1, label_onehot
        loss.backward()
        optimizer.step()
        functional.reset_net(net)
    torch.save(net.state_dict(), "net_train" + ".net")

def test_global(net,device, batch_size, time, dataset_name:str = 'MNIST', encoder:str = 'poisson'):
    with torch.no_grad():
        train_acc = 0
        net = net
        net.load_state_dict(torch.load("net_train" + ".net"))
        functional.set_step_mode(net, 'm')
        loader = get_loader('test',dataset_name, encoder, batch_size, time)
        for img, label in loader:
            img = img.to(device)
            label = label.to(device)
            label_onehot = F.one_hot(label, 10).float()

            out_fr1 = 0.
            img = img.permute(1, 0, 2, 3, 4)
            out_fr1 = net(img)
            out_fr1 = out_fr1.sum(dim=0) / time
            loss = F.mse_loss(out_fr1, label_onehot)
            # print(out_fr1.shape)
            # 正确率的计算方法如下。认为输出层中脉冲发放频率最大的神经元的下标i是分类结果
            train_acc += (out_fr1.argmax(1) == label).float().sum().item()

            # 优化一次参数后，需要重置网络的状态，因为SNN的神经元是有“记忆”的
            functional.reset_net(net)
        print(train_acc / 10000)

def get_loader(phase, dataset_name, encoder, batch_size, time):

    # 创建训练集和测试集数据集实例
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  #可自己更改
    ])
    if dataset_name == 'MNIST':
        Encoder = MNIST_encoder.EncodedMNIST
    elif dataset_name == 'CIFAR10':
        Encoder = Cifar10_encoder.EncodedCIFAR10
    elif dataset_name == 'CIFAR100':
        Encoder = Cifar100_encoder.EncodedCIFAR100
    elif dataset_name == 'FASHIONMNIST':
        Encoder = FashionMNIST_encoder.EncodedFashionMNIST
    train_dataset = Encoder(
        encoding_type=encoder,
        T=time,
        train=True,
        transform=transform  # encoded_images 已经是张量，不需要对其进行额外的转换
    )

    test_dataset = Encoder(
        encoding_type=encoder,
        T=time,
        train=False,
        transform=transform
    )

    # 使用DataLoader加载数据集
    batch_size_ = batch_size
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size_,
        shuffle=True
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size_,
        shuffle=False
    )
    if phase == 'train':
        return train_loader
    else:
        return test_loader



