
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
        x():
        net (Tensor): The tensor of input potentials.
        kwta (int, optional): The number of winners. Default: 1
        inhibition_radius (int, optional): The radius of lateral inhibition. Default: 0
        down():
        time():
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

#简化STDP不根据具体时间大小进行更新，只根据时间的顺序更新
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
def Tstdp_linear_single_step(
    fc: nn.Linear, in_spike: torch.Tensor, out_spike: torch.Tensor,
    trace_pre1: Union[float, torch.Tensor, None],trace_pre2: Union[float, torch.Tensor, None],
    trace_post1: Union[float, torch.Tensor, None],trace_post2: Union[float, torch.Tensor, None],
    tau_pre1: float, tau_pre2: float, tau_post1: float, tau_post2: float,
    f_pre: Callable = lambda x: x, f_post: Callable = lambda x: x
):
    if trace_pre1 is None:
        trace_pre1 = 0.
    if trace_pre2 is None:
        trace_pre2 = 0.
    if trace_post1 is None:
        trace_post1 = 0.
    if trace_post2 is None:
        trace_post2 = 0.

    weight = fc.weight.data
    trace_pre1 = trace_pre1 - trace_pre1 / tau_pre1 + in_spike      # shape = [batch_size, N_in]
    trace_pre2 = trace_pre2 - trace_pre2 / tau_pre2 + in_spike      # shape = [batch_size, N_in]
    trace_post1 = trace_post1 - trace_post1 / tau_post1 + out_spike # shape = [batch_size, N_out]
    trace_post2 = trace_post2 - trace_post2 / tau_post2 + out_spike # shape = [batch_size, N_out]
    
    # [batch_size, N_out, N_in] -> [N_out, N_in]
    delta_w_pre = -f_pre(weight) * (trace_post1.unsqueeze(2) * in_spike.unsqueeze(1)).sum(0) -f_pre(weight) *\
                    (trace_post1.unsqueeze(2) * in_spike.unsqueeze(1)).sum(0) * (trace_pre2.unsqueeze(1) * out_spike.unsqueeze(2)).sum(0)
    delta_w_post = f_post(weight) * (trace_pre1.unsqueeze(1) * out_spike.unsqueeze(2)).sum(0) + f_post(weight) *\
                    (trace_post2.unsqueeze(2) * in_spike.unsqueeze(1)).sum(0) * (trace_pre1.unsqueeze(1) * out_spike.unsqueeze(2)).sum(0)
    return trace_pre1, trace_post1, trace_pre2, trace_post2, delta_w_pre + delta_w_post

def Tstdp_conv2d_single_step(
    conv: nn.Conv2d, in_spike: torch.Tensor, out_spike: torch.Tensor,
    trace_pre1: Union[torch.Tensor, None], trace_pre2: Union[torch.Tensor, None], 
    trace_post1: Union[torch.Tensor, None], trace_post2: Union[torch.Tensor, None],
    tau_pre1: float, tau_pre2: float, 
    tau_post1: float, tau_post2: float,
    f_pre: Callable = lambda x: x, f_post: Callable = lambda x: x
):
    
    if conv.dilation != (1, 1):
        raise NotImplementedError(
            'STDP with dilation != 1 for Conv2d has not been implemented!'
        )
    if conv.groups != 1:
        raise NotImplementedError(
            'STDP with groups != 1 for Conv2d has not been implemented!'
        )

    stride_h = conv.stride[0]
    stride_w = conv.stride[1]

    if conv.padding == (0, 0):
        pass
    else:
        pH = conv.padding[0]
        pW = conv.padding[1]
        if conv.padding_mode != 'zeros':
            in_spike = F.pad(
                in_spike, conv._reversed_padding_repeated_twice,
                mode=conv.padding_mode
            )
        else:
            in_spike = F.pad(in_spike, pad=(pW, pW, pH, pH))

    if trace_pre1 is None:
        trace_pre1 = torch.zeros_like(
            in_spike, device=in_spike.device, dtype=in_spike.dtype
        )
    if trace_pre2 is None:
        trace_pre2 = torch.zeros_like(
            in_spike, device=in_spike.device, dtype=in_spike.dtype
        )
    if trace_post1 is None:
        trace_post1 = torch.zeros_like(
            out_spike, device=in_spike.device, dtype=in_spike.dtype
        )
    if trace_post2 is None:
        trace_post2 = torch.zeros_like(
            out_spike, device = in_spike.device, dtype=in_spike.dtype
        )

    trace_pre1 = trace_pre1 - trace_pre1 / tau_pre1 + in_spike
    trace_post1 = trace_post1 - trace_post1 / tau_post1 + out_spike

    trace_pre2 = trace_pre2 - trace_pre2 / tau_pre2 
    trace_post2 = trace_post2 - trace_post2 / tau_post2 
    
    delta_w = torch.zeros_like(conv.weight.data)
    for h in range(conv.weight.shape[2]):
        for w in range(conv.weight.shape[3]):
            h_end = in_spike.shape[2] - conv.weight.shape[2] + 1 + h
            w_end = in_spike.shape[3] - conv.weight.shape[3] + 1 + w

            pre_spike = in_spike[:, :, h:h_end:stride_h, w:w_end:stride_w]  # shape = [batch_size, C_in, h_out, w_out]
            post_spike = out_spike  # shape = [batch_size, C_out, h_out, h_out]
            weight = conv.weight.data[:, :, h, w]   # shape = [batch_size_out, C_in]

            tr_pre1 = trace_pre1[:, :, h:h_end:stride_h, w:w_end:stride_w]    # shape = [batch_size, C_in, h_out, w_out]
            tr_pre2 = trace_pre2[:, :, h:h_end:stride_h, w:w_end:stride_w]    # shape = [batch_size, C_in, h_out, w_out]
            tr_post1 = trace_post1   # shape = [batch_size, C_out, h_out, w_out]
            tr_post2 = trace_post2   # shape = [batch_size, C_out, h_out, w_out]

            delta_w_pre = - (f_pre(weight) *\
                            (tr_post1.unsqueeze(2) * pre_spike.unsqueeze(1))\
                            .permute([1, 2, 0, 3, 4]).sum(dim = [2, 3, 4])) - (f_pre(weight) * 0.005 *\
                            ((tr_post1.unsqueeze(2) * pre_spike.unsqueeze(1)) * (tr_pre2.unsqueeze(1) * post_spike.unsqueeze(2)))\
                            .permute([1, 2, 0, 3, 4]).sum(dim = [2, 3, 4]))
            delta_w_post = (f_post(weight) *\
                           (tr_pre1.unsqueeze(1) * post_spike.unsqueeze(2))\
                           .permute([1, 2, 0, 3, 4]).sum(dim = [2, 3, 4])) + f_post(weight)  * 0.005 *\
                           ((tr_post2.unsqueeze(2) * pre_spike.unsqueeze(1)) * (tr_pre1.unsqueeze(1) * post_spike.unsqueeze(2)))\
                            .permute([1, 2, 0, 3, 4]).sum(dim = [2, 3, 4])

            delta_w[:, :, h, w] += delta_w_pre + delta_w_post
    trace_pre2 = trace_pre2  + in_spike
    trace_post2 = trace_post2  + out_spike
    return trace_pre1, trace_post1, trace_pre2, trace_post2, delta_w
    
def resume_linear_multi_step(
    fc: nn.Linear, 
    in_spike: torch.Tensor, 
    out_spike: torch.Tensor,
    trace_pre: Union[float, torch.Tensor, None],
    a: float,
    targets: torch.Tensor,
    tau_pre: float
    
):
    weight = fc.weight.data
    delta_w_ = torch.zeros_like(weight)
    T = in_spike.shape[0]

    # [batch_size, N_out, 1] * [batch_size, 1, N_in]  -->[N_out, N_in]
    for t in range(T):
        trace_pre, dw = resume_linear_single_step(
            fc, in_spike[t], out_spike[t], trace_pre, a,targets[t],tau_pre
        )
        delta_w_ += dw
    
    return trace_pre, delta_w_


class resume_Temporal(base.MemoryModule):
    def __init__(self, step_mode: str,
        synapse: Union[nn.Conv2d, nn.Linear], 
        sn: neuron.BaseNode,
        tau_pre: float,
        a: float
):
        super().__init__()
        self.step_mode = step_mode
        self.tau_pre = tau_pre
        self.a = a
        self.synapse = synapse
        self.in_spike_monitor = monitor.InputMonitor(synapse)
        self.out_spike_monitor = monitor.OutputMonitor(sn)
        self.register_memory('trace_pre', None)
        
    def reset(self):
        super(resume_Temporal, self).reset()
        self.in_spike_monitor.clear_recorded_data()
        self.out_spike_monitor.clear_recorded_data()


    def disable(self):
        self.in_spike_monitor.disable()
        self.out_spike_monitor.disable()


    def enable(self):
        self.in_spike_monitor.enable()
        self.out_spike_monitor.enable()
        
        
    def step(self, targets, on_grad: bool = True, scale: float = 1.):
        length = self.in_spike_monitor.records.__len__()
        delta_w = None
        if self.step_mode == 's':
            if isinstance(self.synapse, nn.Linear):
                resume_f = resume_linear_single_step
            elif isinstance(self.synapse, nn.Conv2d):
                stdp_f = stdp_conv2d_single_step
            elif isinstance(self.synapse, nn.Conv1d):
                stdp_f = stdp_conv1d_single_step
            else:
                raise NotImplementedError(self.synapse)
        elif self.step_mode == 'm':
            if isinstance(self.synapse, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                resume_f = resume_linear_multi_step
            else:
                raise NotImplementedError(self.synapse)
        else:
            raise ValueError(self.step_mode)
        for i in range(length):
            
            in_spike = self.in_spike_monitor.records.pop(0)     # [batch_size, N_in]
            out_spike = self.out_spike_monitor.records.pop(0)   # [batch_size, N_out]
            self.trace_pre, dw = resume_f(
                self.synapse, in_spike, out_spike, self.trace_pre,
                self.a, targets, self.tau_pre,
            )
            
            if scale != 1.:
                dw *= scale

            delta_w = dw if (delta_w is None) else (delta_w + dw)

        if on_grad:
            if self.synapse.weight.grad is None:
                self.synapse.weight.grad = -delta_w
            else:
                self.synapse.weight.grad = self.synapse.weight.grad - delta_w
        else:
            return delta_w

class resume(base.MemoryModule):
    def __init__(self, step_mode: str,
        synapse: Union[nn.Conv2d, nn.Linear], 
        sn: neuron.BaseNode,
        tau_pre: float,
        a: float
):
        super().__init__()
        self.step_mode = step_mode
        self.tau_pre = tau_pre
        self.a = a
        self.synapse = synapse
        self.in_spike_monitor = monitor.InputMonitor(synapse)
        self.out_spike_monitor = monitor.OutputMonitor(sn)
        self.register_memory('trace_pre', None)
        
    def reset(self):
        super(resume, self).reset()
        self.in_spike_monitor.clear_recorded_data()
        self.out_spike_monitor.clear_recorded_data()


    def disable(self):
        self.in_spike_monitor.disable()
        self.out_spike_monitor.disable()


    def enable(self):
        self.in_spike_monitor.enable()
        self.out_spike_monitor.enable()
        
        
    def step(self, targets, on_grad: bool = True, scale: float = 1.):
        length = self.in_spike_monitor.records.__len__()
        delta_w = None
        for i in range(length):
            
            in_spike = self.in_spike_monitor.records.pop(0)     # [batch_size, N_in]
            out_spike = self.out_spike_monitor.records.pop(0)   # [batch_size, N_out]

            self.trace_pre, dw = resume_linear_single_step(
                self.synapse, in_spike, out_spike, self.trace_pre,
                self.a, targets, self.tau_pre,
            )
            
            if scale != 1.:
                dw *= scale

            delta_w = dw if (delta_w is None) else (delta_w + dw)

        if on_grad:
            if self.synapse.weight.grad is None:
                self.synapse.weight.grad = -delta_w
            else:
                self.synapse.weight.grad = self.synapse.weight.grad - delta_w
        else:
            return delta_w
            
class TSTDPLearner(base.MemoryModule):
    def __init__(
        self, step_mode: str,
        synapse: Union[nn.Conv2d, nn.Linear], sn: neuron.BaseNode,
        tau_pre1: float, tau_pre2: float, tau_post1: float, tau_post2: float,
        f_pre: Callable = lambda x: x, f_post: Callable = lambda x: x
    ):
        super().__init__()
        self.step_mode = step_mode
        self.tau_pre1 = tau_pre1
        self.tau_post1 = tau_post1
        self.tau_pre2 = tau_pre2
        self.tau_post2 = tau_post2
        self.f_pre = f_pre
        self.f_post = f_post
        self.synapse = synapse
        self.in_spike_monitor = monitor.InputMonitor(synapse)
        self.out_spike_monitor = monitor.OutputMonitor(sn)

        self.register_memory('trace_pre1', None)
        self.register_memory('trace_post1', None)
        self.register_memory('trace_pre2', None)
        self.register_memory('trace_post2', None)

    def reset(self):
        super(TSTDPLearner, self).reset()
        self.in_spike_monitor.clear_recorded_data()
        self.out_spike_monitor.clear_recorded_data()


    def disable(self):
        self.in_spike_monitor.disable()
        self.out_spike_monitor.disable()


    def enable(self):
        self.in_spike_monitor.enable()
        self.out_spike_monitor.enable()


    def step(self, on_grad: bool = True, scale: float = 1.):
        length = self.in_spike_monitor.records.__len__()
        delta_w = None
        if isinstance(self.synapse, nn.Conv2d):
            stdp_f = Tstdp_conv2d_single_step
            #raise NotImplementedError(self.synapse)
        elif isinstance(self.synapse, nn.Linear):
            stdp_f = Tstdp_linear_single_step
        else:
            raise NotImplementedError(self.synapse)
        for _ in range(length):
            in_spike = self.in_spike_monitor.records.pop(0)     # [batch_size, N_in]
            out_spike = self.out_spike_monitor.records.pop(0)   # [batch_size, N_out]

            self.trace_pre1, self.trace_post1, self.trace_pre2, self.trace_post2, dw = stdp_f(
                self.synapse, in_spike, out_spike,
                self.trace_pre1, self.trace_pre1, self.trace_post1, self.trace_post2, 
                self.tau_pre1, self.tau_pre2, self.tau_post1, self.tau_post2,
                self.f_pre, self.f_post
            )
            if scale != 1.:
                dw *= scale

            delta_w = dw if (delta_w is None) else (delta_w + dw)

        if on_grad:
            if self.synapse.weight.grad is None:
                self.synapse.weight.grad = -delta_w
            else:
                self.synapse.weight.grad = self.synapse.weight.grad - delta_w
        else:
            return delta_w
