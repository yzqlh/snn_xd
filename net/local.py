import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.svm import LinearSVC
import torch.nn.functional as F
from spikingjelly.activation_based import functional,neuron, monitor, base, layer, STBP, learning
import torch.nn as nn
from spikingjelly.Dataset_encoding import MNIST_encoder, Cifar10_encoder, Cifar100_encoder, FashionMNIST_encoder, encoding
import numpy as np
import math
import os
import logging
###从此开始
#这部分代码为高斯滤波器和次序编码
class FilterKernel:
    def __init__(self, window_size):
        self.window_size = window_size

    def __call__(self):
        pass

class DoGKernel(FilterKernel):
    def __init__(self, window_size, sigma1, sigma2):
        super(DoGKernel, self).__init__(window_size)
        self.sigma1 = sigma1
        self.sigma2 = sigma2

    def __call__(self):
        w = self.window_size // 2
        x, y = np.mgrid[-w: w + 1: 1, -w: w + 1: 1]
        a = 1.0 / (2 * math.pi)
        prod = x ** 2 + y ** 2
        f1 = (1 / (self.sigma1 ** 2)) * \
             np.exp(-0.5 * (1 / (self.sigma1 ** 2)) * prod)
        f2 = (1 / (self.sigma2 ** 2)) * \
             np.exp(-0.5 * (1 / (self.sigma2 ** 2)) * prod)
        dog = a * (f1 - f2)
        dog = (dog - np.mean(dog)) / np.max(dog)
        dog_tensor = torch.from_numpy(dog)
        return dog_tensor.float()

class Filter:
    def __init__(self, filter_kernels, padding=0, threshold=None):
        self.max_window_size = filter_kernels[0].window_size
        self.kernels = torch.stack([kernel().unsqueeze(0)
                                    for kernel in filter_kernels])
        self.number_of_kernels = len(filter_kernels)
        self.padding = padding
        self.threshold = threshold

    def __call__(self, input):
        output = F.conv2d(input, self.kernels, padding=self.padding).float()
        output = torch.where(output < self.threshold, torch.tensor(
            0.0, device=output.device), output)
        return output

def local_normalization(input, normalization_radius, eps=1e-12):
    length = normalization_radius * 2 + 1
    kernel = torch.ones(1, 1, length, length,
                        device=input.device).float() / ((length) ** 2)
    y = input.squeeze(0)
    y.unsqueeze_(1)
    means = F.conv2d(y, kernel, padding=normalization_radius) + eps
    y = y / means
    y.squeeze_(1)
    y.unsqueeze_(0)
    return y

class S1Transform:
    def __init__(self, filter1, timesteps=15):
        self.to_tensor = transforms.ToTensor()
        self.to_gray = transforms.Grayscale()
        self.filter = filter1
        self.temporal_transform = encoding.Intensity2Latency(
            timesteps, to_spike=True)
        self.cnt = 1

    def __call__(self, image):
        if self.cnt % 10000 == 0:
            logging.info(f'Preprocessed {self.cnt} images')
        self.cnt += 1
        image = self.to_tensor(image) * 255
        image = self.to_gray(image)
        image = image.unsqueeze(0)
        image = self.filter(image)
        image = local_normalization(image, 8)

        temporal_image = self.temporal_transform(image)
        return temporal_image
###到此为止

def pad(input, pad, value=0):
    return F.pad(input, pad, value=value)

class inhibit_layer(nn.Module):
    def __init__(self):
        super(inhibit_layer, self).__init__()
    def forward(self, x):
        if self.training:
            x = x.squeeze(1)
            x = learning.pointwise_inhibition(x)
            x = x.unsqueeze(1)
        else:
            pass
        return x

class MaxPool2d_1(nn.Module):
    def __init__(self):
        super(MaxPool2d_1, self).__init__()
    def forward(self,x):

        x = x.permute(1,2,3,4,0)#[B,C,W,H,T]
        device = x.device
        x1 = functional.first_spike_index(x).to(device)
        x1 = x1.float()
        x = x1.permute(4,0,1,2,3)
        return x
'''
class Last_MaxPool2d(nn.Module):
    def __init__(self,layer):
        super(Last_MaxPool2d, self).__init__()
        self.layer = layer
    def forward(self,x):
        if self.training:
            pass
        else:
            h = model[self.rank].h  # [T,B,C,W,H]
            x = self.net[self.layer].h
            x = x.squeeze(1)
            pooled_spk, _ = torch.max(x.reshape(x.size(1), -1), dim=1)
            print(pooled_spk)
            spk_out = pooled_spk.view(1, x.size(1))
            x = spk_out
            print(torch.sum(x==pooled_spk))
        return x
'''

def pass_through_network(model, loader, device='cuda'):
    X_path = 'tmp/test_x.npy'
    y_path = 'tmp/test_y.npy'
    if os.path.isfile(X_path):
        features = np.load(X_path)
        targets = np.load(y_path)
    else:
        os.makedirs('tmp', exist_ok=True)
        features = []
        targets = []
        # num=0
        for data, target in loader:
            # num +=1
            features.append(pass_batch_through_network(model, data, device))
            targets.append(target)
            # print(num)
        features = np.concatenate(features)
        targets = np.concatenate(targets)
        np.save(X_path, features)
        np.save(y_path, targets)
    return features, targets


def pass_batch_through_network(model, batch, device='cuda'):
    with torch.no_grad():
        ans = []
        for data in batch:
            data_in = data.to(device)
            data_in = data_in.unsqueeze(1)
            output = model(data_in)
            h = model[6].h
            h = h.squeeze(1)
            pooled_spk, _ = torch.max(h.reshape(h.size(1), -1), dim=1)
            spk_out = pooled_spk.view(1, h.size(1))
            h = spk_out
            functional.reset_net(model)
            ans.append(h.reshape(-1).cpu().numpy())
        return np.array(ans)


def eval(X, y, predictions):
    non_silence_mask = np.count_nonzero(X, axis=1) > 0
    correct_mask = predictions == y
    correct_non_silence = np.logical_and(correct_mask, non_silence_mask)
    correct = np.count_nonzero(correct_non_silence)
    silence = np.count_nonzero(~non_silence_mask)
    return (correct / len(X), (len(X) - (correct + silence)) / len(X), silence / len(X))


def train_STDP(device, batch_size, time, dataset_name:str = 'MNIST'):

    net = nn.Sequential(
            layer.Conv2d(2, 30, kernel_size=5, padding=2, bias=False),
            neuron.IFNode(v_threshold=15.),
            layer.MaxPool2d(kernel_size=2, stride=2,padding=1),
            MaxPool2d_1(),
            inhibit_layer(),
            layer.Conv2d(30, 100, kernel_size=5, padding=1, bias=False),
            neuron.IFNode(v_threshold=1000.),
            ).to(device)
    conv1_kwinners = 5
    conv1_inhibition_rad = 2
    conv2_kwinners = 8
    conv2_inhibition_rad = 1
    functional.set_step_mode(net, 'm')
    loader = get_loader('train', dataset_name, batch_size, time)

    #train_layer(0,net,loader,device,conv1_kwinners,conv1_inhibition_rad)

    net.load_state_dict(torch.load('model_directory' + "saved_l" + str(0) + ".net"))
    # train_layer(5,net,loader,device,conv2_kwinners,conv2_inhibition_rad)
    net.load_state_dict(torch.load('model_directory' + "saved_l" + str(5) + ".net"))
    #print(list(net[0].weight.data))


    train_eval_classifier(net, loader, device, C=2.4)


#逐层训练代码
def train_layer(num_layer, net, loader, device, conv1_kwinners, conv1_inhibition_rad):
    net.train()
    nn.init.normal_(net[num_layer].weight.data, mean=0.8, std=0.05)
    #print(list(net[num_layer].weight.data))
    with torch.no_grad():
        spk_cnt = 0
        learning_convergence = 1
        num = 1
        max_ap = torch.Tensor([0.15]).to(device)
        in_spike_monitor = monitor.InputMonitor(net[num_layer])
        out_spike_monitor = monitor.OutputMonitor(net[num_layer + 1])
        stdp = learning.STDP(conv_layer=net[num_layer])
        net_path = 'model_directory' + "saved_l" + str(num_layer) + ".net"
        if os.path.isfile(net_path):
            net.load_state_dict(torch.load(net_path))
        else:
            while learning_convergence > 0.01:
                print(
                    f"======================== Epoch {num} ========================")
                print(
                    f"======================== Layer {num_layer} ========================")
                print(
                    f'======================== Convergence {learning_convergence} ====================')
                for index, data in enumerate(loader, 0):
                    img, label = data
                    for i in range(len(img)):

                        if num_layer == 0:
                            spk_cnt += 1
                            if spk_cnt >= 500:
                                spk_cnt = 0
                                ap = stdp.learning_rate[0] \
                                         .clone().detach() \
                                         .to(device) * 2
                                ap = torch.min(ap, max_ap)
                                an = ap * -0.75
                                stdp.update_learning_rate(ap, an)
                            img1 = img[i].to(device)
                            img1 = img1.unsqueeze(0)
                            img1 = img1.permute(1, 0, 2, 3, 4).to(device)

                            out = net(img1)

                            potential = net[num_layer + 1].h
                            potential[potential < 15.] = 0.

                            potential = potential.squeeze(1)
                            pot = learning.pointwise_inhibition(potential)
                            spk = pot.sign()
                            winners = learning.get_k_winners(pot, conv1_kwinners, conv1_inhibition_rad, spk)
                            #print(winners)
                            #如果卷积层padding不等于0则在监视器中应该同卷积层padding做一样的操作
                            if net[num_layer].padding[0] != 0:
                                num_padding = net[num_layer].padding[0]
                                in_spike = pad(in_spike_monitor.records.pop(0).squeeze(1), (num_padding, num_padding, num_padding, num_padding))
                            else:
                                in_spike = in_spike_monitor.records.pop(0).squeeze(1)

                            stdp(in_spike, spk, winners)

                        else:
                            img1 = img[i].to(device)
                            img1 = img1.unsqueeze(0)
                            img1 = img1.permute(1, 0, 2, 3, 4).to(device)
                            out = net(img1)

                            potential = net[num_layer + 1].h
                            potential[potential < 10.] = 0.
                            potential = potential.squeeze(1)

                            pot = learning.pointwise_inhibition(potential)
                            spk = pot.sign()

                            winners = learning.get_k_winners(pot, conv1_kwinners, conv1_inhibition_rad, spk)
                            #print(torch.sum(out))
                            if net[num_layer].padding[0] != 0:
                                num_padding = net[num_layer].padding[0]
                                in_spike = pad(in_spike_monitor.records.pop(0).squeeze(1), (num_padding, num_padding, num_padding, num_padding))
                            else:
                                in_spike = in_spike_monitor.records.pop(0).squeeze(1)
                            stdp(in_spike, spk, winners)


                        in_spike_monitor.clear_recorded_data()
                        out_spike_monitor.clear_recorded_data()
                        functional.reset_net(net)
                num += 1
                weights = net[num_layer].weight.data
                learning_convergence = calculate_learning_convergence(weights)
                torch.save(net.state_dict(), net_path)
            print(
                f"===========================================================================")
            print(
                f"======================== Training layer {num_layer} complete ========================")
            print(
                f"===========================================================================")
            torch.save(net.state_dict(), net_path)


#取数据集
def get_loader(phase, dataset_name, batch_size, time):

    kernels1 = [DoGKernel(7, 1, 2), DoGKernel(7, 2, 1)]
    filter1 = Filter(kernels1, padding=3, threshold=50)
    s1_transform = S1Transform(filter1,time)
    if dataset_name == 'MNIST':
        Encoder = datasets.MNIST
    elif dataset_name == 'CIFAR10':
        Encoder = datasets.CIFAR10
    elif dataset_name == 'CIFAR100':
        Encoder = datasets.CIFAR100

    train_dataset = Encoder(
        root='D:/Jupyter/data',
        train=True,
        transform=s1_transform,
        download=True
    )

    test_dataset = Encoder(
        root='D:/Jupyter/data',
        train=False,
        transform=s1_transform,
        download=True
    )

    train_data_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=0,
        pin_memory=True
    )

    test_data_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=True
    )

    if phase == 'train':
        return train_data_loader
    else:
        return test_data_loader

#判断训练是否完成的函数
def calculate_learning_convergence(weights):
    n_w = weights.numel()
    sum_wf_i = torch.sum(weights * (1 - weights))
    c_l = sum_wf_i / n_w
    return c_l.item()

def train_eval_classifier(model, loader, device, C=2.4, max_iter=1000):

    print('Training the classifier...')
    pt_path = 'clf.pt'

    # setting the model to prediction mode
    model.train()
    train_X, train_y = pass_through_network(
        model, loader, device)

    clf = LinearSVC(C=C, dual=False)
    clf.fit(train_X, train_y)
    torch.save(clf, pt_path)
    predictions = clf.predict(train_X)
    accuracy, error, silence = eval(train_X, train_y, predictions)
    print(accuracy)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_STDP(device=device, batch_size=1, time=15,)
