import cv2 as cv
from torchvision import transforms
from encoding import PoissonEncoder, LatencyEncoder, Rank_order_Encoder
from typing import List, Optional
import torch
from matplotlib import pyplot as plt
import numpy as np

img = cv.imread('D:/test coding/000001.png')
# print("img.shape:", img.shape)   # numpy数组格式为（H,W,C）

transf = transforms.ToTensor()
img_tensor = transf(img)  # tensor数据格式是torch(C,H,W)

T_poi = 20
poi_en = PoissonEncoder()
poi_out = poi_en(img_tensor)
for t in range(T_poi):
    # 在每个时间步调用poi_en(x)
    print("PoissonEncoder:", poi_out)


T_lat = 800
lat_en = LatencyEncoder(T_lat)
lat_out = lat_en(img_tensor)
# for t in range(T_lat):
#     print("LatencyEncoder:", lat_out)


T_rank = 28*28
rank_en = Rank_order_Encoder(T_rank)
rank_out = rank_en(img_tensor)
# for t in range(T_rank):
#     print("Rank_order_Encoder:", rank_out)



def plot_spike_trains(
    spikes: torch.Tensor,
    n_ex: Optional[int] = None,
    top_k: Optional[int] = None,
    indices: Optional[List[int]] = None,
) -> None:
    # language=rst
    """
    Plot spike trains for top-k neurons or for specific indices.

    :param spikes: Spikes for one simulation run of shape
        ``(n_examples, n_neurons, time)``.
    :param n_ex: Allows user to pick which example to plot spikes for.
    :param top_k: Plot k neurons that spiked the most for n_ex example.
    :param indices: Plot specific neurons' spiking activity instead of top_k.
    """
    assert n_ex is not None and 0 <= n_ex < spikes.shape[0]

    plt.figure()

    if top_k is None and indices is None:  # Plot all neurons' spiking activity
        spike_per_neuron = [np.argwhere(i == 1).flatten() for i in spikes[n_ex, :, :]]
        plt.title("Spiking activity for all %d neurons" % spikes.shape[1])

    elif top_k is None:  # Plot based on indices parameter
        assert indices is not None
        spike_per_neuron = [
            np.argwhere(i == 1).flatten() for i in spikes[n_ex, indices, :]
        ]

    elif indices is None:  # Plot based on top_k parameter
        assert top_k is not None
        # Obtain the top k neurons that fired the most
        top_k_loc = np.argsort(np.sum(spikes[n_ex, :, :], axis=1), axis=0)[::-1]
        spike_per_neuron = [
            np.argwhere(i == 1).flatten() for i in spikes[n_ex, top_k_loc[0:top_k], :]
        ]
        plt.title("Spiking activity for top %d neurons" % top_k)

    else:
        raise ValueError('One of "top_k" or "indices" or both must be None')

    plt.eventplot(spike_per_neuron, linelengths=[0.5] * len(spike_per_neuron))
    plt.xlabel("Simulation Time")
    plt.ylabel("Neuron index")
    plt.show()


# #####################################################################泊松编码#####################################################################

T_poi = 20
poi_en = PoissonEncoder()

poi_list = []
for t in range(T_poi):
    # 在每个时间步调用poi_en(x)并更新poi_list
    poi_out = poi_en(img_tensor).unsqueeze(0)
    poi_list.append(poi_out)

# 合并所有时间步的脉冲输出
poi_list = torch.cat(poi_list, 0)
#
poi_out_flat = poi_list.view(poi_list.shape[0], poi_list.shape[1], -1)

d_anti_seq = list(range(1, poi_out_flat.ndim))
d_anti_seq.insert(2, poi_out_flat.ndim-3)
poi_out_ter = poi_out_flat.permute(d_anti_seq)


for n_ex in range(poi_out_ter.shape[0]):
    poi_visual = plot_spike_trains(poi_out_ter, n_ex)

# #####################################################################延迟编码#####################################################################
#
# T_lat = 800
# lat_en = LatencyEncoder(T_lat)
#
# lat_list = []
# for t in range(T_lat):
#     # 在每个时间步调用lat_en(x)并更新lat_list
#     lat_out = lat_en(img_tensor).unsqueeze(0)
#     lat_list.append(lat_out)

# 合并所有时间步的脉冲输出
# lat_list = torch.cat(lat_list, 0)
# lat_out_flat = lat_list.view(lat_list.shape[0], lat_list.shape[1], -1)
#
# d_anti_seq = list(range(1, lat_out_flat.ndim))
# d_anti_seq.insert(2, lat_out_flat.ndim-3)
# lat_one_hot = lat_out_flat.permute(d_anti_seq)
#
#
# for n_ex in range(lat_one_hot.shape[0]):
#     lat_visual = plot_spike_trains(lat_one_hot, n_ex)

# #####################################################################次序编码#####################################################################
#
# T_rank = 784
# rank_en = Rank_order_Encoder(T_rank)
#
# rank_list = []
# for t in range(T_rank):
#     # 在每个时间步调用rank_en(x)并更新rank_list
#     rank_out = rank_en(img_tensor).unsqueeze(0)
#     rank_list.append(rank_out)
#
#
# # 合并所有时间步的脉冲输出
# rank_list = torch.cat(rank_list, 0)
# print("rank_list:", rank_list)
# print("rank_list.size:", rank_list.size())
#
# rank_out_flat = rank_list.view(rank_list.shape[0], rank_list.shape[1], -1)
#
# d_anti_seq = list(range(1, rank_out_flat.ndim))
# d_anti_seq.insert(2, rank_out_flat.ndim-3)
# rank_one_hot = rank_out_flat.permute(d_anti_seq)
#
#
# for n_ex in range(rank_one_hot.shape[0]):
#     rank_visual = plot_spike_trains(rank_one_hot, n_ex)
