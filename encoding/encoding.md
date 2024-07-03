# encoding  
## *class* encoding.PoissonEncoder(step_mode='s')  
**基类:** StatelessEncoder  
无状态的泊松编码器。输出脉冲的发放概率与输入 x 相同。  
必须确保 0 <= x <= 1。  
***
## *class* encoding.LatencyEncoder(T: int, enc_function='linear', step_mode='s')  
**基类：** StatefulEncoder  
**参数:**  
- T (int) – 最大（最晚）脉冲发放时刻  
- enc_function (str) – 定义使用哪个函数将输入强度转化为脉冲发放时刻，可以为 linear 或 log  
延迟编码器，将 0 <= x <= 1 的输入转化为在 0 <= t_f <= T-1 时刻发放的脉冲。输入的强度越大，发放越早。   
当 enc_function == 'linear'  
t_f(x) = (T - 1)(1 - x)  
当 enc_function == 'log'  
t_f(x) = (T - 1) - ln(\\alpha * x + 1)  
其中 Math input error 满足 t_f(1) = T - 1
***
## *class* encoding.Rank_order_Encoder(T:int, time: int, step_mode='s')  
**基类:** StatefulEncoder  
**参数:**   
- x – 形状的张量  
- time(int) – 每个输入变量的秩序编码尖峰序列的长度  
- T(int) – 模拟时间步长  
次序编码器，抛弃精确的时间信息，只取决于神经元脉冲发放的顺序，按输入强度的大小转化神经元发放脉冲的顺序，强度最大的输入对应的神经元最先发放脉冲，以此类推。  