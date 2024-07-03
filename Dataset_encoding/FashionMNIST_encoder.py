from torch.utils.data import Dataset
from torchvision import datasets, transforms
import torch
from encoding import PoissonEncoder, LatencyEncoder, Rank_order_Encoder
from torch.utils.data import DataLoader
import os

class EncodedFashionMNIST(Dataset):
    def __init__(self,  encoding_type, T, transform, train=True, target_transform=None):
        self.train = train
        self.encoding_type = encoding_type
        self.T = T
        self.transform = transform
        self.target_transform = target_transform
        self.encoder = self.get_encoder(encoding_type)
        self.data, self.labels = self.get_encoded_images_and_labels()
    def get_encoder(self, encoding_type):
        if encoding_type == 'poisson':
            self.encoder = PoissonEncoder()
        elif encoding_type == 'latency':
            self.encoder = LatencyEncoder()
        elif encoding_type == 'rank':
            self.encoder = Rank_order_Encoder()
        else:
            raise ValueError(f"Unknown encoding type: {encoding_type}")
        return self.encoder

    def get_encoded_images_and_labels(self):
        # 加载原始FashionMNIST数据集
        dataset_ = datasets.FashionMNIST(
            root='data/',
            train=self.train,
            download=True,
            transform=transforms.ToTensor(),
        )
        # 自动创建 spikedata 文件夹及子文件夹 FashionMNIST_encoded
        encoded_folder_name = "FashionMNIST_encoded"
        spikedata_folder = "spikedata"
        encoded_data_folder = os.path.join(spikedata_folder, encoded_folder_name)
        os.makedirs(encoded_data_folder, exist_ok=True)  # 如果不存在，将创建这些文件夹

        # 定义编码类型字符串
        encoding_type_str = {
            'poisson': 'poisson',
            'latency': 'latency',
            'rank': 'rank'
        }.get(self.encoding_type, 'unknown')
        # 根据训练集或测试集构造文件名

        batch_size = 10000
        num_batches = len(dataset_) // batch_size
        data = []
        labels = []
        for batch_idx in range(num_batches):

            dataset_str = 'train' if self.train else 'test'
            save_name = f"batch{batch_idx + 1}_encoded_images_{encoding_type_str}_T{self.T}_{dataset_str}_Fmnist.pt"
            encoded_images_path = os.path.join(encoded_data_folder, save_name)
            save_name_labels = f"batch{batch_idx + 1}_labels_{encoding_type_str}_T{self.T}_{dataset_str}_Fmnist.pt"
            labels_path = os.path.join(encoded_data_folder, save_name_labels)

            # 检查文件是否已存在
            if os.path.exists(encoded_images_path) and os.path.exists(labels_path):
                encoded_images = torch.load(encoded_images_path)
                batch_labels = torch.load(labels_path)
                data.append(encoded_images)
                labels.append(batch_labels)
            else:
                batch_start_idx = batch_idx * batch_size
                batch_end_idx = min(batch_start_idx + batch_size, len(dataset_))
                encoded_images = []
                batch_labels = []
                for idx in range(batch_start_idx, batch_end_idx):
                    image, label = dataset_[idx]
                    print(f"Encoding Batch Index: {idx}")
                    en_list = []
                    for t in range(self.T):
                        en_out = self.encoder(image)
                        en_list.append(en_out.unsqueeze(0))
                    en_list = torch.cat(en_list, 0)
                    encoded_images.append(en_list.unsqueeze(1))
                    labels_tensor = torch.tensor(label, dtype=torch.long).unsqueeze(0)
                    batch_labels.append(labels_tensor)
                # 将列表转换为张量
                encoded_images = torch.cat(encoded_images, 1)
                batch_labels = torch.cat(batch_labels)

                # 保存编码后的脉冲数据集到"spikedata"文件夹
                torch.save(encoded_images, encoded_images_path)
                torch.save(batch_labels, labels_path)

                data.append(encoded_images)
                labels.append(batch_labels)

        return data, labels

    def __getitem__(self, index):
        batch_size = 10000
        batch_idx = index // batch_size
        within_batch_idx = index % batch_size

        if batch_idx >= len(self.data):
            raise IndexError(
                f"Index {index} out of range for data dimension 1 with size {len(self.labels) * batch_size}")

        image = self.data[batch_idx][:, within_batch_idx, :, :, :]
        label = self.labels[batch_idx][within_batch_idx]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label


    def __len__(self):
        return sum(len(batch) for batch in self.labels)


# 创建训练集和测试集数据集实例
train_dataset = EncodedFashionMNIST(
    encoding_type='poisson',
    T=20,
    train=True,
    transform=None  #encoded_images 已经是张量，不需要对其进行额外的转换
)

test_dataset = EncodedFashionMNIST(
    encoding_type='poisson',
    T=20,
    train=False,
    transform=None
)

# 使用DataLoader加载数据集
batch_size = 64
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False
)

# 迭代DataLoader
for loader in (train_loader, test_loader):
    for batch_idx, (images, labels) in enumerate(loader):
        print(f"Batch Index: {batch_idx}")
        print("images.shape", images.shape)
        print("labels.shape", labels.shape)

        # 此处可以添加模型训练或测试代码