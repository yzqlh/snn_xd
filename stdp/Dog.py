import matplotlib.pyplot as plt
import custuom_transform
import torch
device = 'cuda'
num_classes = 10
batchsize = 640
train_dataset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Grayscale(1)]))


import torch
import matplotlib.pyplot as plt
image,y = train_dataset[0]
image = image.squeeze().numpy()
##image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

# 对原始图像应用第一组高斯平滑参数
smoothed1 = cv2.GaussianBlur(image, (5, 5), 1)

# 对第一次平滑后的图像应用第二组高斯平滑参数
smoothed2 = cv2.GaussianBlur(image, (9, 9), 0)


smoothed3 = cv2.GaussianBlur(image, (7, 7), 2)

# 计算正向高斯差分
dog_positive = smoothed1 - smoothed2
dog_positive[dog_positive<0]=0
# 计算反向高斯差分
dog_negative = smoothed1 - smoothed3

dog_negative_tensor = torch.tensor(dog_negative)
dog_positive_tensor = torch.tensor(dog_positive)

plt.imshow(dog_positive_tensor, cmap='gray')
plt.show()
