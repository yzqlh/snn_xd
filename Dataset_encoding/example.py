from torch.utils.data import DataLoader
from MNIST_encoder import EncodedMNIST
from FashionMNIST_encoder import EncodedFashionMNIST
from Cifar10_encoder import EncodedCIFAR10
from Cifar100_encoder import EncodedCIFAR100



##############MNIST################
# 创建训练集和测试集数据集实例
train_dataset = EncodedMNIST(
    encoding_type='poisson',
    T=20,
    train=True,
    transform=None  #encoded_images 已经是张量，不需要对其进行额外的转换
)

test_dataset = EncodedMNIST(
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
        # 此处可以添加模型训练或测试代码


##############FashionMNIST################
# 创建训练集和测试集数据集实例
train_dataset = EncodedFashionMNIST(
    train=True,
    encoding_type='poisson',
    T=10,
    transform=None  #encoded_images 已经是张量，不需要对其进行额外的转换
)

test_dataset = EncodedFashionMNIST(
    train=False,
    encoding_type='poisson',
    T=10,
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
        # 此处可以添加模型训练或测试代码


# ##############Cifar10################
# # 选择编码器类型并实例化
# encoding_type = 'poisson'  # 可以是 'poisson', 'latency', 或 'rank'
# encoder = SpikingEncoder(encoding_type)
#
# # 创建训练集和测试集数据集实例
# train_dataset = EncodedCIFAR10(
#     train=True,
#     encoder=encoder,
#     T=20,
#     transform=None  #encoded_images 已经是张量，不需要对其进行额外的转换
# )
#
# test_dataset = EncodedCIFAR10(
#     train=False,
#     encoder=encoder,
#     T=20,
#     transform=None
# )
#
# # 使用DataLoader加载数据集
# batch_size = 64
# train_loader = DataLoader(
#     dataset=train_dataset,
#     batch_size=batch_size,
#     shuffle=True
# )
#
# test_loader = DataLoader(
#     dataset=test_dataset,
#     batch_size=batch_size,
#     shuffle=False
# )
#
# # 迭代DataLoader
# for loader in (train_loader, test_loader):
#     for batch_idx, (images, labels) in enumerate(loader):
#         print(f"Batch Index: {batch_idx}")
#         # 此处可以添加模型训练或测试代码
#
#
#
# ##############Cifar100################
# # 选择编码器类型并实例化
# encoding_type = 'poisson'  # 可以是 'poisson', 'latency', 或 'rank'
# encoder = SpikingEncoder(encoding_type)
#
# # 创建训练集和测试集数据集实例
# train_dataset = EncodedCIFAR100Dataset(
#     train=True,
#     encoder=encoder,
#     T=20,
#     transform=None  #encoded_images 已经是张量，不需要对其进行额外的转换
# )
#
# test_dataset = EncodedCIFAR100Dataset(
#     train=False,
#     encoder=encoder,
#     T=20,
#     transform=None
# )
#
# # 使用DataLoader加载数据集
# batch_size = 64
# train_loader = DataLoader(
#     dataset=train_dataset,
#     batch_size=batch_size,
#     shuffle=True
# )
#
# test_loader = DataLoader(
#     dataset=test_dataset,
#     batch_size=batch_size,
#     shuffle=False
# )
#
# # 迭代DataLoader
# for loader in (train_loader, test_loader):
#     for batch_idx, (images, labels) in enumerate(loader):
#         print(f"Batch Index: {batch_idx}")
#         # 此处可以添加模型训练或测试代码
