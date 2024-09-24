目前该网络只支持离线数据集
## net

- STBP全局学习算法网络实现
- STDP局部学习算法网络实现
### **global**

#### train_global
```python
def train_global(net, device, batch_size, time, dataset_name:str = 'MNIST', encoder:str = 'poisson'):
```
- net：用户定义的网络结构
- device：启用GPU加速定义的device
- batch_size：一次训练多少数据
- time：时间步长
- dataset_name：所训练的数据集的名称，默认为`MNIST`
- encoder：所使用的数据集编码器，默认为`poisson`

#### test_global
```python
def train_global(net, device, batch_size, time, dataset_name:str = 'MNIST', encoder:str = 'poisson'):
```
- net：用户定义的网络结构
- device：启用GPU加速定义的device
- batch_size：一次训练多少数据
- time：时间步长
- dataset_name：所训练的数据集的名称，默认为`MNIST`
- encoder：所使用的数据集编码器，默认为`poisson`

#### get_loader
```python
def get_loader(phase, dataset_name, encoder, batch_size, time):
```
- phase：测试阶段或训练阶段
- dataset_name：所训练的数据集的名称，默认为`MNIST`
- encoder：所使用的数据集编码器，默认为`poisson`
- batch_size：一次训练多少数据
- time：时间步长

#### example
```python
net =nn.Sequential(
        layer.Conv2d(1,10,5,bias=False),
        STBP.LIFSpike(),
        layer.MaxPool2d(2),
        layer.Conv2d(10,20,5,bias=False),
        STBP.LIFSpike(),
        layer.MaxPool2d(2),
        layer.Conv2d(20,40,5,padding = 2,bias=False),
        STBP.LIFSpike(),
        layer.MaxPool2d(2),
        layer.Flatten(),
        layer.Linear(40*2*2, 10, bias=False),
        STBP.LIFSpike(),

        ).to(device)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_global(net,device=device, batch_size=60, time=30,)
test_global(net,device=device, batch_size=60, time=30,)
```
