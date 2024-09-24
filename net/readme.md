## net

### **global**
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
