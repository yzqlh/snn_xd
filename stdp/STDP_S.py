
from spikingjelly.activation_based import base,layer,neuron,functional,learning,encoding,surrogate,monitor
import torch.nn as nn
import torch,gc
import torch.nn.functional as F
import argparse
import torch.utils.data as data
from torch.cuda import amp
from torchvision import transforms
from torchvision import datasets
import cv2
import numpy as np
import math

def get_k_winners(x,net,kwta=1,inhibition_radius=0,down=24,time=30):
    r"""Finds at most :attr:`kwta` winners first based on the earliest spike time, then based on the maximum potential.
    It returns a list of winners, each in a tuple of form (feature, row, column).

    .. note::

        Winners are selected sequentially. Each winner inhibits surrounding neruons in a specific radius in all of the
        other feature maps. Note that only one winner can be selected from each feature map.

    Args:
        potentials (Tensor): The tensor of input potentials.
        kwta (int, optional): The number of winners. Default: 1
        inhibition_radius (int, optional): The radius of lateral inhibition. Default: 0
        spikes (Tensor, optional): Spike-wave corresponding to the input potentials. Default: None

    Returns:
        List: List of winners.
    """
    h = net
    x = x.permute(1,2,3,4,0)                        #对输入脉冲进行重构以便后续操作
    time = x.shape[4]                               #提取时间步长
    tongdao_num = x.shape[1]                        #提取通道长度
    w_num = x.shape[2]                              #提取长度和宽度
    device = x.device                               #提取device
    h = h.permute(1,2,3,4,0).to(device)
    x1 = functional.first_spike_index(x)            #找到发出首脉冲的神经元索引
    x1 = (x1.int()) * (torch.arange(0,time).detach().to(device) + 1)  #对x1进行操作以便可以通过min得到首脉冲
    x1[x1 == 0] = time + 1                          #对于无脉冲的情况0将其改成31
    tongdao_seq = {}
    x_h_position_seq = {}
    x_w_position_seq = {}
    num = 0
    for i in range(kwta):
        x_min = x1.min()
        if x_min > 25:
            break
        num += 1
        h_max = h[...,x_min-1].max()
        idx = torch.where(h[...,x_min-1]==h_max)

        x_h_position = idx[3][0]
        x_w_position = idx[2][0]
        tongdao = idx[1][0]
        tongdao_seq[i] = tongdao
        x_h_position_seq[i] = x_h_position
        x_w_position_seq[i] = x_w_position
         #每个通道应采取不同的神经元
        x_h_position_left = x_h_position - inhibition_radius
        x_h_position_right = x_h_position + inhibition_radius
        x_w_position_up = x_w_position - inhibition_radius
        x_w_position_down = x_w_position + inhibition_radius
        
        if x_h_position_left < 0:
            x_h_position_left = 0
        if x_w_position_up < 0:
            x_w_position_up = 0
        if x_h_position_right > down - 1:
            x_h_position_right = down - 1
        if x_w_position_down > down - 1:
            x_w_position_down = down - 1
        if tongdao == 0:
            h[:,tongdao+1:,x_w_position_up:x_w_position_down+1,x_h_position_left:x_h_position_right+1,:] = 0
            x1[:,0:tongdao,x_w_position_up:x_w_position_down+1,x_h_position_left:x_h_position_right+1,:] = time + 2
            x1[:,tongdao+1:,x_w_position_up:x_w_position_down+1,x_h_position_left:x_h_position_right+1,:] = time + 2
        else:
            h[:,0:tongdao,x_w_position_up:x_w_position_down+1,x_h_position_left:x_h_position_right+1,:] = 0
            h[:,tongdao+1:,x_w_position_up:x_w_position_down+1,x_h_position_left:x_h_position_right+1,:] = 0
            x1[:,0:tongdao,x_w_position_up:x_w_position_down+1,x_h_position_left:x_h_position_right+1,:] = time + 2
            x1[:,tongdao+1:,x_w_position_up:x_w_position_down+1,x_h_position_left:x_h_position_right+1,:] = time + 2
        h[:,tongdao,:,:,:] = 0
        x1[:,tongdao,:,:,:] = time + 2
    winner = [(tongdao_seq[i],x_w_position_seq[i],x_h_position_seq[i]) for i in range(num)] 
    return winner

def pointwise_inhibition(spike,thresholded_potentials):
    
    spike = spike.permute(1,2,3,4,0)
    _,tongdao_num,w_num,_,time = spike.shape
    device = spike.device                               #提取device
    thresholded_potentials = thresholded_potentials.permute(1,2,3,4,0).to(device)
    x1 = functional.first_spike_index(spike)            #找到发出首脉冲的神经元索引
    x1 = (x1.int()) * (torch.arange(0,time).to(device) + 1)  #对x1进行操作以便可以通过min得到首脉冲
    x1[x1 == 0] = time + 1                          #对于无脉冲的情况0将其改成31
    x_min,x_idx = x1.min(dim=4)
    x_min_,x_idx_ = x_min.min(dim=1)
    x_min_[x_min_==time+1] = time+2
    
    for i in range(tongdao_num):
        thresholded_potentials[:,i,~(torch.eq(x_min[:,i,...],x_min_).squeeze(0)),:] = 0
    for i in range(time):
        if torch.all(thresholded_potentials[...,i]==0):
            pass
        else:
            thresholded_potentials_max,thresholded_potentials_idx = thresholded_potentials[...,i].max(dim=1)
            thresholded_potentials_idx[thresholded_potentials_max==0] = -1
            thresholded_potentials_idx = thresholded_potentials_idx.repeat(tongdao_num,1,1)
            for j in range(tongdao_num):
                # 创建一个与原始张量形状相同的全零张量  
                mask = torch.zeros_like(thresholded_potentials_idx[j]) 
                mask[thresholded_potentials_idx[j]==j] = 1
                mask = mask.bool()
                if j == 0:
                    thresholded_potentials[:,j+1:,mask,:] = 0.
                else:
                    thresholded_potentials[:,0:j,mask,:] = 0.
                    thresholded_potentials[:,j+1:,mask,:] = 0.
    thresholded_potentials = thresholded_potentials.permute(4,0,1,2,3)
    return thresholded_potentials

class STDP(nn.Module):
    def __init__(self, conv_layer, learning_rate=(0.004, -0.003), use_stabilizer=True, lower_bound=0, upper_bound=1):
        super(STDP, self).__init__()
        self.conv_layer = conv_layer
        self.learning_rate = (torch.tensor([learning_rate[0]]),
                              torch.tensor([learning_rate[1]]))
        self.use_stabilizer = use_stabilizer
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        

    def get_pre_post_ordering(self, input_spikes, output_spikes, winners):
        r"""Computes the ordering of the input and output spikes with respect to the position of each winner and
        returns them as a list of boolean tensors. True for pre-then-post (or concurrency) and False for post-then-pre.
        Input and output tensors must be spike-waves.

        Args:
            input_spikes (Tensor): Input spike-wave
            output_spikes (Tensor): Output spike-wave
            winners (List of Tuples): List of winners. Each tuple denotes a winner in a form of a triplet (feature, row, column).

        Returns:
            List: pre-post ordering of spikes
        """

        # accumulating input and output spikes to get latencies
        input_spikes = input_spikes.permute(1,2,3,0) * (torch.arange(0,input_spikes.shape[0]).detach().to(input_spikes.device) + 1)
        input_spikes = input_spikes.permute(3,0,1,2)
        
        input_latencies = torch.sum(input_spikes, dim=0)
        input_latencies[input_latencies==0] = 10000000
        
        output_spikes = output_spikes.permute(1,2,3,0)
        output_spikes = functional.first_spike_index(output_spikes) 
       
        output_spikes = output_spikes * (torch.arange(0,output_spikes.shape[3]).detach().to(output_spikes.device) + 1)
        output_spikes = output_spikes.permute(3,0,1,2)
        
        output_latencies = torch.sum(output_spikes, dim=0)
        
        result = []
        for winner in winners:
            # generating repeated output tensor with the same size of the receptive field
            out_tensor = torch.ones(
                *self.conv_layer.kernel_size, device=output_latencies.device) * output_latencies[winner]
            #print(output_latencies[winner])
            # slicing input tensor with the same size of the receptive field centered around winner
            # since there is no padding, there is no need to shift it to the center
            in_tensor = input_latencies[:, winner[-2]: winner[-2] + self.conv_layer.kernel_size[-2],
                                        winner[-1]: winner[-1] + self.conv_layer.kernel_size[-1]]
            #print(out_tensor)
            result.append(torch.le(in_tensor, out_tensor))

        return result

    def forward(self, input_spikes, output_spikes, winners):
        pairings = self.get_pre_post_ordering(
            input_spikes, output_spikes, winners)

        lr = torch.zeros_like(self.conv_layer.weight)
        for i in range(len(winners)):
            winner = winners[i][0]
            pair = pairings[i].clone().detach().to(device)
            lr0 = self.learning_rate[0].clone().detach().to(device)
            lr1 = self.learning_rate[1].clone().detach().to(device)
            lr[winner.item()] = torch.where(pair, lr0, lr1)

        self.conv_layer.weight += lr * ((self.conv_layer.weight - self.lower_bound) * (
            self.upper_bound - self.conv_layer.weight) if self.use_stabilizer else 1)
        self.conv_layer.weight.clamp_(self.lower_bound, self.upper_bound)

    def update_learning_rate(self, ap, an):
        self.learning_rate = tuple([ap, an])
