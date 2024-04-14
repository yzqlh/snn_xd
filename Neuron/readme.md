目前默认单步训练，训练需要自行添加时间维度T

## Base

- 记忆模块的基类，设置记录神经元记录膜电压、阈值等状态
- 后期设置单步与多步的基类
- base中定义了神经元被调用时默认执行 single_step_forward 函数

## **surrogate**

梯度替代函数类

### Sigmoid

- sigmoid_backward：定义替代函数
- sigmoid：修改向前向后传播方式

## Neuron

### **BaseNode**

```python
(v_threshold: float = 1., v_reset: float = 0., surrogate_function:** **Callable** **surrogate.Sigmoid(), step_mode='s')
```

此Node为所有神经元的基类

- **v_threshold** (float) – 神经元的阈值电压
- tau (float)  神经元膜电位的时间常量，默认为6.0
- **v_reset** (float) – 神经元的重置电压。如果不为 `None`，当神经元释放脉冲后，电压会被重置为 `v_reset`； 如果设置为 `None`，当神经元释放脉冲后，电压会被减去 `v_threshold`
- **surrogate_function** (*Callable*) – 反向传播时用来计算脉冲函数梯度的替代函数
- **step_mode** (str) – 步进模式，可以为 ‘s’ (单步) 或 ‘m’ (多步)
- ~~**store_v_seq** (bool) – 在使用 `step_mode = 'm'` 时，给与 `shape = [T, N, *]` 的输入后，是否保存中间过程的 `shape = [T, N, *]` 的各个时间步的电压值 `self.v_seq` 。设置为 `False` 时计算完成后只保留最后一个时刻的电压，即 `shape = [N, *]` 的 `self.v` 。 通常设置成 `False` ，可以节省内存~~

#### 核心子函数

1. neuronal_charge：膜电压的积累（神经元的充电差分方程）
2. neuronal_fire：根据当前神经元的电压、阈值，计算输出*脉冲*
3. neuronal_reset：对膜电位进行重置
   1. Hard方式：释放脉冲后，膜电位直接被设置成重置电压: $$V= V_{reset}$$
   2. Soft方式：释放脉冲后，膜电位减去阈值电压：$$V = V - V_{th}$$

### **IFNode**

```python
(v_threshold: float = 1., v_reset: float = 0., surrogate_function:** **Callable** **surrogate.Sigmoid(), step_mode='s')
```

- **v_threshold** (float) – 神经元的阈值电压
- **v_reset** (float) – 神经元的重置电压。如果不为 `None`，当神经元释放脉冲后，电压会被重置为 `v_reset`； 如果设置为 `None`，当神经元释放脉冲后，电压会被减去 `v_threshold`
- **surrogate_function** (*Callable*) – 反向传播时用来计算脉冲函数梯度的替代函数

#### 核心子函数

1. neuronal_charge：$$V = V + X$$
2. jit_eval_single_step_forward_hard(soft)_reset：计算神经元膜电压与激发，即：neuronal_charge + neuronal_fire函数

### **LIFNode**

```python
(tau: float= 2.0, decay_input: bool = True,v_threshold: float= 1.0, v_reset: float= 0.0,surrogate_function: Callable = Sigmoid(alpha=4.0, spiking=True)
```

- **tau** (float) – 膜电位时间常数
- **decay_input** (bool) – 输入是否也会参与衰减
  - **decay_input** = true
    - $$V = V_{t-1}+ \frac{1}{\tau} (x_t-(v_{t-1}-V_{reset})) $$
  - **decay_input** = false
    - $$V = V_{t-1}+ \frac{1}{\tau} (V_{t-1}-V_{reset}) +X_t$$
- **v_threshold** (float) – 神经元的阈值电压
- **v_reset** (float) – 神经元的重置电压。如果不为 `None`，当神经元释放脉冲后，电压会被重置为 `v_reset`； 如果设置为 `None`，当神经元释放脉冲后，电压会被减去 `v_threshold`
- **surrogate_function** (*Callable*) – 反向传播时用来计算脉冲函数梯度的替代函数

#### 核心子函数

1. ##### jit_eval_single_step_forward_hard_reset_decay_input：计算神经元膜电压与激发，**v_reset** =hard + **decay_input** =true

2. ##### jit_eval_single_step_forward_hard_reset_no_decay_input：计算神经元膜电压与激发**，v_reset** =hard + **decay_input** =false

3. ##### jit_eval_single_step_forward_soft_reset_decay_input：**计算神经元膜电压与激发，v_reset** =soft + **decay_input** =true

4. ##### jit_eval_single_step_forward_soft_reset_no_decay_input：计算神经元膜电压与激发，v_reset** =soft+ decay_input =false

### SRM

```python
(self, tau_m=10, tau_s=5, v_threshold: float = 1., v_reset: float = 0.,
             surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False, step_mode='s')
```

$$ V(t) = V_{\text{rest}} + (V_{\text{rest}} - V(t-1)) \times \exp\left(-\frac{\Delta t}{\tau_m}\right) + \sum_{i=0}^{N} \frac{w_i}{\tau_s} \times \exp\left(-\frac{\Delta t - t_i}{\tau_s}\right) $$​

- $ V(t) $：时间 \( t \) 时刻的膜电压
- $V_{\text{rest}} $：静息电位
- $ \Delta t $：当前时刻与上一个时刻的时间间
- $ \tau_m $：膜时间常数，描述了膜电压的恢复速度
-  N ：突触输入的脉冲数
- $ w_i $：每个脉冲的突触权重
- $ t_i $：第 \( i \) 个脉冲的时刻

使用了近似的形式，忽略了膜电压的漏电流部分中的 $\exp\left(-\frac{\Delta t - t_i}{\tau_s}\right)$部分，直接在每个时间步长内按比例更新膜电压。突触电流部分的计算使用了类似的形式，按时间间隔和突触权重的比例叠加。