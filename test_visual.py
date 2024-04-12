import torch
import cv2 as cv
from torchvision import transforms
from encoding import PoissonEncoder, LatencyEncoder, Rank_order_Encoder
from typing import List, Optional

import torch
from matplotlib import pyplot as plt
import numpy as np

img = cv.imread('D:/test coding/000001.png')
print("img.shape:", img.shape)   # numpy数组格式为（H,W,C）

transf = transforms.ToTensor()
img_tensor = transf(img)  # tensor数据格式是torch(C,H,W)
print("img_tensor.shape:", img_tensor.shape)



poi_en = PoissonEncoder()
poi_out = poi_en(img_tensor)
print("PoissonEncoder:", poi_out)
print("poi_out.shape:", poi_out.shape)



T_lat = 800
lat_en = LatencyEncoder(T_lat)
lat_out = lat_en(img_tensor)
print("lat_out:", lat_out)
print("lat_out.shape", lat_out.shape)
for t in range(T_lat):
    print("LatencyEncoder:", lat_out)


T_rank = 3*28*28
rank_en = Rank_order_Encoder(T_rank)
rank_out = rank_en(img_tensor)
print(rank_out)
# for t in range(T):
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



# poi_out_flat = reshaped = poi_out.view(poi_out.shape[0], poi_out.shape[1]*poi_out.shape[2], 1)
# print("poi_out_flat.shape:", poi_out_flat.shape)
# poi_out_transformed = poi_out_flat.repeat(1, 1, 10)
#
# for n_ex in range(poi_out_transformed .shape[0]):
#     Poi_visual = plot_spike_trains(poi_out_transformed, n_ex)

# lat_en = LatencyEncoder(T_lat)
# lat_origin = lat_en.single_step_encode(img_tensor)
# print("lat_origin.shape", lat_origin.shape)
#

lat_list = torch.empty(0, lat_out.shape[0], lat_out.shape[1]*lat_out.shape[2])
# print("lat_list.shape:", lat_list.shape)
for t in range(T_lat):
    lat_out_flat = reshaped = lat_out.view(1, lat_out.shape[0], lat_out.shape[1]*lat_out.shape[2])
    # print("lat_out_flat:", lat_out_flat)
    print("lat_out.shape:", lat_out.shape)
    print("lat_out_flat.shape:", lat_out_flat.shape)
    lat_list = torch.cat((lat_list, lat_out_flat), dim = 0)
print("lat_list.shape:", lat_list.shape)
# print(lat_list)
d_anti_seq = list(range(1, lat_list.ndim))
d_anti_seq.insert(2, lat_list.ndim -3)
lat_one_hot = lat_list.permute(d_anti_seq)
print("lat_one_hot.shape:", lat_one_hot.shape)
print("lat_one_hot:", lat_one_hot)

# for n_ex in range(lat_one_hot.shape[0]):
#     lat_visual = plot_spike_trains(lat_one_hot, n_ex)





# lat_out_flat = reshaped = lat_out.view(lat_out.shape[0], lat_out.shape[1]*lat_out.shape[2])
# print("lat_out_flat_shape:", lat_out_flat.shape)
# # 在第二个维度上插入一个新的维度，大小为 1
# lat_out_expanded = lat_out_flat.unsqueeze(2)
# print("lat_out_expanded_shape:", lat_out_expanded.shape)
# # 将张量沿着指定的维度进行扩展，使其形状变为 [3, 784, 280]
# lat_out_transformed = lat_out_expanded.repeat(1, 1, T_lat)
# print("lat_out_transformed_shape:", lat_out_transformed.shape)
# for n_ex in range(lat_out_transformed.shape[0]):
#     lat_visual = plot_spike_trains(lat_out_transformed, n_ex)
#
#
# rank_out_flat = reshaped = rank_out.view(poi_out.shape[0], poi_out.shape[1]*poi_out.shape[2], T_rank/3)
# for n_ex in range(rank_out.shape[0]):
#     print(n_ex)
#     rank_visual = plot_spike_trains(rank_out_flat, n_ex)