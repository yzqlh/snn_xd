## STBP
- 定义了脉冲激活类 SpikeAct：用于梯度近似
- 定义了脉冲迭代神经元LIFSpike
- 定义了state_update函数：用于LIFSpike中的膜电位更新并应用SpikeAct

### **SpikeAct**

```python
class SpikeAct(torch.autograd.Function):
```

#### 核心子函数

- forward：实现前向传播，并保存前向传播中的张量
- backward：利用梯度近似的方式实现反向传播

### **state_update**

```python
def state_update(u_t_n1, o_t_n1, W_mul_o_t1_n):
```
- u_t_n1：上一时刻的膜电位
- o_t_n1：上一时刻的输出
- W_mul_o_t1_n：上一时刻的输出与权重乘积

### **LIFSpike**

```python
(steps=STEPS)
```

- STEPS：时间步长
