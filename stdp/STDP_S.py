from spikingjelly.datasets.n_mnist import NMNIST
from torch.utils.data import DataLoader
from spikingjelly.activation_based import base,layer,neuron,functional,learning,encoding,surrogate,monitor
import torch.nn as nn
import torch,gc

from spikingjelly.datasets import play_frame
import torch.nn.functional as F
from snntorch import functional as SF
import argparse
import torch.nn.utils.weight_norm as weight_norm
import torch.utils.data as data
from torch.cuda import amp
from torchvision import transforms
from torchvision import datasets
from matplotlib import pyplot as plt
import cv2
import numpy as np
import math


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
