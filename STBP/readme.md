## STBP
- 定义了脉冲激活函数 SpikeAct：用于梯度近似
- 定义了脉冲迭代神经元LIFSpike
- 定义了state_update：用于LIFSpike中的膜电位更新并应用SpikeAct

### **BaseNode**

```python
(v_threshold: float = 1., v_reset: float = 0., surrogate_function:** **Callable** **surrogate.Sigmoid(), step_mode='s')
```
