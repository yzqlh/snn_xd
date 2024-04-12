
import cv2 as cv
from torchvision import transforms, datasets
from encoding import PoissonEncoder, LatencyEncoder, Rank_order_Encoder


img = cv.imread('D:/test coding/000001.png')
print(img.shape)   # numpy数组格式为（H,W,C）

transf = transforms.ToTensor()
img_tensor = transf(img)  # tensor数据格式是torch(C,H,W)
print(img_tensor.size())


poi_en = PoissonEncoder()
poi_out = poi_en(img_tensor)
print("PoissonEncoder:", poi_out)
print("poi_out.shape:", poi_out.shape)
#
T = 800
lat_en = LatencyEncoder(T)
for t in range(T):
    print("LatencyEncoder:", lat_en(img_tensor))


T =2400
rank_en = Rank_order_Encoder(T)
for t in range(T):
    print("Rank_order_Encoder:", rank_en(img_tensor))


# rank_en = Rank_order_Encoder(T = 20 , time = 20)
# rank_out = rank_en(img_tensor)
# print("Rank_order_Encoder:", rank_out)

# sub_images = []
# for i in range(0, img_tensor.size()[0], 1):
#     for j in range(0, img_tensor.size()[1], 28):
#         for k in range(0, img_tensor.size()[2], 28):
#             # 提取 256x256 大小的子图
#             sub_image = img_tensor[i:i+1, j:j + 28, k:k + 28]
#
#             # 如果子图大小不足 28*28，则进行补零操作
#             if sub_image.size(1) < 28 or sub_image.size(2) < 28:
#                 sub_image = torch.nn.functional.pad(sub_image, (0, 28 - sub_image.size(2), 0, 28 - sub_image.size(1)))
#                 # 将分割后的子图保存到列表中
#                 sub_images.append(sub_image)
#
# rank_en = Rank_order_Encoder(T = 1, time = 28*28)
#
# # 将分割后的子图输入到编码器中
# rank_outs = []
# for sub_image in sub_images:
#     rank_out = rank_en(sub_image.unsqueeze(0))  # 假设编码器输出批次为单个子图
#     rank_outs.append(rank_out)
#
#
# for i, rank_out in enumerate(rank_outs):
#
#     print("Rank_order_Encoder:", rank_outs)
