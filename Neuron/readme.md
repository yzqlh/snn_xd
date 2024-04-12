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

**(v_threshold: float = 1., v_reset: float = 0., surrogate_function:** **Callable** **surrogate.Sigmoid(), step_mode='s')**

此Node为所有神经元的基类

- **v_threshold** (float) – 神经元的阈值电压
- tau ([float)  神经元膜电位的时间常量，默认为6.0
- **v_reset** (float) – 神经元的重置电压。如果不为 `None`，当神经元释放脉冲后，电压会被重置为 `v_reset`； 如果设置为 `None`，当神经元释放脉冲后，电压会被减去 `v_threshold`
- **surrogate_function** (*Callable*) – 反向传播时用来计算脉冲函数梯度的替代函数
- **step_mode** (str) – 步进模式，可以为 ‘s’ (单步) 或 ‘m’ (多步)
- **store_v_seq** (bool) – 在使用 `step_mode = 'm'` 时，给与 `shape = [T, N, *]` 的输入后，是否保存中间过程的 `shape = [T, N, *]` 的各个时间步的电压值 `self.v_seq` 。设置为 `False` 时计算完成后只保留最后一个时刻的电压，即 `shape = [N, *]` 的 `self.v` 。 通常设置成 `False` ，可以节省内存

#### 子函数

1. neuronal_charge：膜电压的积累（*神经元**的充电差分方程*）
2. neuronal_fire：*根据当前**神经元**的电压、阈值，计算输出**脉冲*
3. neuronal_reset：*对膜**电位**进行重置*
   1. Hard方式：释放脉冲后，膜电位直接被设置成重置电压: $$V= V_{reset}$$
   2. Soft方式：释放脉冲后，膜电位减去阈值电压：$$V = V - V_{th}$$

### **IFNode**

**(v_threshold: float = 1., v_reset: float = 0., surrogate_function:** **Callable** **surrogate.Sigmoid(), step_mode='s')**

- **v_threshold** (float) – 神经元的阈值电压
- **v_reset** (float) – 神经元的重置电压。如果不为 `None`，当神经元释放脉冲后，电压会被重置为 `v_reset`； 如果设置为 `None`，当神经元释放脉冲后，电压会被减去 `v_threshold`
- **surrogate_function** (*Callable*) – 反向传播时用来计算脉冲函数梯度的替代函数

#### 核心子函数

1. neuronal_charge：$$V = V + X$$
2. jit_eval_single_step_forward_hard(soft)_reset：计算神经元膜电压与激发，即：neuronal_charge + neuronal_fire函数

### **LIFNode**

(tau: float= 2.0***,** ***decay_input:*** bool = True,v_threshold: float= 1.0*****,** ***v_reset:***  float= 0.0****,surrogate_function: Callable ***= Sigmoid(alpha=4.0, spiking=True)***

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

3. ##### jit_eval_single_step_forward_soft_reset_decay_input：**计算****神经元膜电压与激发，v_reset** =soft + **decay_input** =true

4. ##### jit_eval_single_step_forward_soft_reset_no_decay_input：

**计算****神经元****膜电压与激发，v_reset** =soft+ **decay_input** =false

### SRM

假设神经元在$$t^{-}$$发放脉冲，当此后没有电流输入时，神经元膜电势逐渐回落到静息电位

外部电流输入后，经过κ核函数滤波（视为一种影响），当膜电势增加到阈值时，神经元发放脉冲，反馈两个影响：提升阈值，以及做一个超极化到静息电位的过程。

$$\eta (s) = - \eta_{0} * \exp(\frac{- s}{\tau_{recov}})$$

$$\epsilon(s) = \frac{1}{1-\frac{\tau_s}{\tau_m}}(\exp(\frac{-s}{\tau_m})-\exp(\frac{-s}{\tau_s}))$$