## STBP
- 定义了脉冲激活函数 SpikeAct：用于梯度近似
- 定义了脉冲迭代神经元LIFSpike
- 定义了state_update：用于LIFSpike中的膜电位更新并应用SpikeAct

### **SpikeAct**

```python
class SpikeAct(torch.autograd.Function):
```
