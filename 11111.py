import torch

# 定义要放入的张量数量
num_tensors = 10

# 创建一个空列表，用于存放要连接的张量
tensor_list = []

# 循环遍历，生成十个形状为 [3, 784, 1] 的张量，并加入列表
for _ in range(num_tensors):
    tensor = torch.randn(3, 5, 1)  # 创建一个形状为 [3, 784, 1] 的张量
    tensor_list.append(tensor)

# 使用 torch.cat 方法将张量列表在第三个维度上连接起来
A = torch.cat(tensor_list, dim=2)

# 查看结果张量的形状
print(A.shape)  # 输出 torch.Size([3, 784, 10])
print(A)