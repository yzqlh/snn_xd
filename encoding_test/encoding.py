import torch

class Intensity2Latency:
    def __init__(self, timesteps, to_spike=False):
        self.timesteps = timesteps
        self.to_spike = to_spike

    def transform(self, intensities):
        bins_intensities = []
        nonzero_cnt = torch.nonzero(intensities).size()[0]
        bin_size = nonzero_cnt // self.timesteps
        intensities_flattened = torch.reshape(intensities, (-1,))
        intensities_flattened_sorted = torch.sort(
            intensities_flattened, descending=True)
        sorted_bins_value, sorted_bins_idx = torch.split(
            intensities_flattened_sorted[0], bin_size), torch.split(intensities_flattened_sorted[1], bin_size)
        spike_map = torch.zeros_like(intensities_flattened_sorted[0])
        for i in range(self.timesteps):
            spike_map.scatter_(0, sorted_bins_idx[i], sorted_bins_value[i])
            spike_map_copy = spike_map.clone().detach()
            spike_map_copy = spike_map_copy.reshape(tuple(intensities.shape))
            bins_intensities.append(spike_map_copy.squeeze(0).float())
        return torch.stack(bins_intensities)

    def __call__(self, image):
        if self.to_spike:
            x1 = self.transform(image).sign()
            x1 = x1.permute(1,2,3,0)
            x1 = functional.first_spike_index(x1)
            x1 = x1.permute(3,0,1,2)
            
            return x1.float()
        return self.transform(image)
