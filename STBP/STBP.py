from spikingjelly.activation_based import base,layer,neuron,functional,learning,encoding,surrogate,monitor
import torch.nn as nn
import torch,gc
from spikingjelly.datasets import play_frame
import torch.nn.functional as F
import argparse
import torch.nn.utils.weight_norm as weight_norm
import torch.utils.data as data
from torch.cuda import amp
from torchvision import transforms
from torchvision import datasets
import numpy as np
import math
import os

...
STEPS = 2                           # 时间步数
DT = 5                              # 时间步长，仅在时序（DVS等）数据集下有意义
SIMWIN = DT * STEPS # 仿真时间窗口
ALPHA = 0.5                         # 梯度近似项
VTH = 0.2                           # 阈值电压 V_threshold
TAU = 0.25                          # 漏电常数 tau
...
class SpikeAct(torch.autograd.Function):
    """ 定义脉冲激活函数，并根据论文公式进行梯度的近似。
        Implementation of the spiking activation function with an approximation of gradient.
    """
    alpha = ALPHA

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        # if input = u > Vth then output = 1
        output = torch.gt(input, 0) 
        return output.float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors 
        grad_input = grad_output.clone()
        # hu is an approximate func of df/du
        hu = abs(input) < SpikeAct.alpha
        hu = hu.float() / (2 * SpikeAct.alpha)
        return grad_input * hu


def state_update(u_t_n1, o_t_n1, W_mul_o_t1_n):
    u_t1_n1 = TAU * u_t_n1 * (1 - o_t_n1) + W_mul_o_t1_n
    o_t1_n1 = SpikeAct.apply(u_t1_n1 - VTH)
    return u_t1_n1, o_t1_n1

class LIFSpike(nn.Module):
    """对带有时间维度的张量进行一次LIF神经元的发放模拟，可以视为一个激活函数，用法类似ReLU。
        Generates spikes based on LIF module. It can be considered as an activation function and is used similar to ReLU. The input tensor needs to have an additional time dimension, which in this case is on the last dimension of the data.
    """
    def __init__(self, steps=STEPS):
        super(LIFSpike, self).__init__()
        self.steps = steps

    def forward(self, x):
        u   = torch.zeros(x.shape, device=x.device)
        out = torch.zeros(x.shape, device=x.device)
        u, out= state_update(u, out, x)
        return out
