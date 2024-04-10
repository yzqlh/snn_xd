import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import base
from abc import abstractmethod

class StatelessEncoder(nn.Module, base.StepModule):
    def __init__(self, step_mode='s'):
        """
        * :ref:`API in English <StatelessEncoder.__init__-en>`

        .. _StatelessEncoder.__init__-cn:

        无状态编码器的基类。无状态编码器 ``encoder = StatelessEncoder()``，直接调用 ``encoder(x)`` 即可将 ``x`` 编码为 ``spike``。

        * :ref:`中文API <StatelessEncoder.__init__-cn>`

        .. _StatelessEncoder.__init__-en:

        The base class of stateless encoder. The stateless encoder ``encoder = StatelessEncoder()`` can encode ``x`` to
        ``spike`` by ``encoder(x)``.

        """
        super().__init__()
        self.step_mode = step_mode

    @abstractmethod
    def forward(self, x: torch.Tensor):
        """
        * :ref:`API in English <StatelessEncoder.forward-en>`

        .. _StatelessEncoder.forward-cn:

        :param x: 输入数据
        :type x: torch.Tensor
        :return: ``spike``, shape 与 ``x.shape`` 相同
        :rtype: torch.Tensor

        * :ref:`中文API <StatelessEncoder.forward-cn>`

        .. _StatelessEncoder.forward-en:

        :param x: input data
        :type x: torch.Tensor
        :return: ``spike``, whose shape is same with ``x.shape``
        :rtype: torch.Tensor
        """
        raise NotImplementedError


class StatefulEncoder(base.MemoryModule):
    def __init__(self, T: int, step_mode='s'):
        """
        * :ref:`API in English <StatefulEncoder.__init__-en>`

        .. _StatefulEncoder.__init__-cn:

        :param T: 编码周期。通常情况下，与SNN的仿真周期（总步长一致）
        :type T: int

        有状态编码器的基类。有状态编码器 ``encoder = StatefulEncoder(T)``，编码器会在首次调用 ``encoder(x)`` 时对 ``x`` 进行编码。在第 ``t`` 次调用 ``encoder(x)`` 时会输出 ``spike[t % T]``

        .. code-block:: python

            encoder = StatefulEncoder(T)
            s_list = []
            for t in range(T):
                s_list.append(encoder(x))  # s_list[t] == spike[t]

        * :ref:`中文API <StatefulEncoder.__init__-cn>`

        .. _StatefulEncoder.__init__-en:

        :param T: the encoding period. It is usually same with the total simulation time-steps of SNN
        :type T: int

        The base class of stateful encoder. The stateful encoder ``encoder = StatefulEncoder(T)`` will encode ``x`` to
        ``spike`` at the first time of calling ``encoder(x)``. It will output ``spike[t % T]``  at the ``t`` -th calling

        .. code-block:: python

            encoder = StatefulEncoder(T)
            s_list = []
            for t in range(T):
                s_list.append(encoder(x))  # s_list[t] == spike[t]

        """
        super().__init__()
        self.step_mode = step_mode
        assert isinstance(T, int) and T >= 1
        self.T = T
        self.register_memory('spike', None)
        self.register_memory('t', 0)


    def single_step_forward(self, x: torch.Tensor = None):
        """
        * :ref:`API in English <StatefulEncoder.forward-en>`

        .. _StatefulEncoder.forward-cn:

        :param x: 输入数据
        :type x: torch.Tensor
        :return: ``spike``, shape 与 ``x.shape`` 相同
        :rtype: torch.Tensor

        * :ref:`中文API <StatefulEncoder.forward-cn>`

        .. _StatefulEncoder.forward-en:

        :param x: input data
        :type x: torch.Tensor
        :return: ``spike``, whose shape is same with ``x.shape``
        :rtype: torch.Tensor
        """

        if self.spike is None:
            self.single_step_encode(x)

        t = self.t
        self.t += 1
        if self.t >= self.T:
            self.t = 0
        return self.spike[t]


    @abstractmethod
    def single_step_encode(self, x: torch.Tensor):
        """
        * :ref:`API in English <StatefulEncoder.single_step_encode-en>`

        .. _StatefulEncoder.single_step_encode-cn:

        :param x: 输入数据
        :type x: torch.Tensor
        :return: ``spike``, shape 与 ``x.shape`` 相同
        :rtype: torch.Tensor

        * :ref:`中文API <StatefulEncoder.single_step_encode-cn>`

        .. _StatefulEncoder.single_step_encode-en:

        :param x: input data
        :type x: torch.Tensor
        :return: ``spike``, whose shape is same with ``x.shape``
        :rtype: torch.Tensor
        """
        raise NotImplementedError


    def extra_repr(self) -> str:
        return f'T={self.T}'


class PoissonEncoder(StatelessEncoder):
    def __init__(self, step_mode='s'):
        """
        * :ref:`API in English <PoissonEncoder.__init__-en>`

        .. _PoissonEncoder.__init__-cn:

        无状态的泊松编码器。输出脉冲的发放概率与输入 ``x`` 相同。

        .. warning::

            必须确保 ``0 <= x <= 1``。

        * :ref:`中文API <PoissonEncoder.__init__-cn>`

        .. _PoissonEncoder.__init__-en:

        The poisson encoder will output spike whose firing probability is ``x``。

        .. admonition:: Warning
            :class: warning

            The user must assert ``0 <= x <= 1``.
        """
        super().__init__(step_mode)

    def forward(self, x: torch.Tensor):
        out_spike = torch.rand_like(x).le(x).to(x)
        return out_spike

class LatencyEncoder(StatefulEncoder):

    def __init__(self, T: int, enc_function='linear', step_mode='s'):
        """
        * :ref:`API in English <LatencyEncoder.__init__-en>`

        .. _LatencyEncoder.__init__-cn:

        :param T: 最大（最晚）脉冲发放时刻
        :type T: int
        :param enc_function: 定义使用哪个函数将输入强度转化为脉冲发放时刻，可以为 `linear` 或 `log`
        :type enc_function: str

        延迟编码器，将 ``0 <= x <= 1`` 的输入转化为在 ``0 <= t_f <= T-1`` 时刻发放的脉冲。输入的强度越大，发放越早。
        当 ``enc_function == 'linear'``
            .. math::
                t_f(x) = (T - 1)(1 - x)

        当 ``enc_function == 'log'``
            .. math::
                t_f(x) = (T - 1) - ln(\\alpha * x + 1)

        其中 :math:`\alpha` 满足 :math:`t_f(1) = T - 1`


        实例代码：

        .. code-block:: python

            x = torch.rand(size=[8, 2])
            print('x', x)
            T = 20
            encoder = LatencyEncoder(T)
            for t om range(T):
                print(encoder(x))

        .. warning::

            必须确保 ``0 <= x <= 1``。

        .. warning::

            不要忘记调用reset，因为这个编码器是有状态的。


        * :ref:`中文API <LatencyEncoder.__init__-cn>`

        .. _LatencyEncoder.__init__-en:

        :param T: the maximum (latest) firing time
        :type T: int
        :param enc_function: how to convert intensity to firing time. `linear` or `log`
        :type enc_function: str

        The latency encoder will encode ``0 <= x <= 1`` to spike whose firing time is ``0 <= t_f <= T-1``. A larger
        ``x`` will cause a earlier firing time.

        If ``enc_function == 'linear'``
            .. math::
                t_f(x) = (T - 1)(1 - x)

        If ``enc_function == 'log'``
            .. math::
                t_f(x) = (T - 1) - ln(\\alpha * x + 1)

        where :math:`\alpha` satisfies :math:`t_f(1) = T - 1`


        Example:
        .. code-block:: python

            x = torch.rand(size=[8, 2])
            print('x', x)
            T = 20
            encoder = LatencyEncoder(T)
            for t in range(T):
                print(encoder(x))

        .. admonition:: Warning
            :class: warning

            The user must assert ``0 <= x <= 1``.

        .. admonition:: Warning
            :class: warning

            Do not forget to reset the encoder because the encoder is stateful!

        """
        super().__init__(T, step_mode)
        if enc_function == 'log':
            self.alpha = math.exp(T - 1.) - 1.
        elif enc_function != 'linear':
            raise NotImplementedError

        self.enc_function = enc_function


    def single_step_encode(self, x: torch.Tensor):
        if self.enc_function == 'log':
            t_f = (self.T - 1. - torch.log(self.alpha * x + 1.)).round().long()
        else:
            t_f = ((self.T - 1.) * (1. - x)).round().long()

        self.spike = F.one_hot(t_f, num_classes=self.T).to(x)
        # [*, T] -> [T, *]
        d_seq = list(range(self.spike.ndim - 1))
        d_seq.insert(0, self.spike.ndim - 1)
        self.spike = self.spike.permute(d_seq)


class Rank_order_Encoder(StatefulEncoder):

    def __init__(self , T:int, time: int, step_mode='s'):

        super().__init__(T, step_mode)
        self.time = time

    def single_step_encode(self, x: torch.Tensor):


        """
        Encodes data via a rank order coding-like representation. One spike per neuron,
        temporally ordered by decreasing intensity. Inputs must be non-negative.

        :param x: Tensor of shape ``[n_samples, n_1, ..., n_k]``.
        :param time: Length of rank order-encoded spike train per input variable.
        :param T: Simulation time step.
        :return: Tensor of shape ``[time, n_1, ..., n_k]`` of rank order-encoded spikes.
        首先，断言输入数据 X 非负，如果不满足条件，则抛出异常。
        然后，函数获取输入数据的形状和元素总数，并将其展平为一维张量。
        接着，函数通过将输入数据除以其最大值来创建尖峰时间，并通过乘以模拟时间来扩展尖峰时间。
        之后，函数创建一个形状为 [time, size] 的零张量 spikes，用于存储尖峰时间。
        依次寻times中最大值,获得索引,在 spikes 张量的对应位置标记为1。
        最后，函数将 spikes 张量重新塑形为 [time, n_1, ..., n_k] 的形状，并返回这个张量。
        示例代码:

        .. code-block:: python

            time = 5
            input_data=[0.5, 0.3, 0.8, 0.2, 0.6]
            x_in = torch.Tensor(input_data)

            # 创建 Rank_order_Encoder 类的实例
            encoder = Rank_order_Encoder(T=1, time=time)

            # 进行秩序编码并打印结果
            binary_code = encoder.single_step_encode(x = x_in)
            print("Binary code for input signal:", binary_code)

        """
        assert (x >= 0).all(), "Inputs must be non-negative"

        shape, size = x.shape, x.numel()
        x = x.flatten()
        time = int(self.time / self.T)

        # Create spike times in order of decreasing intensity.
        x /= x.max()
        times = torch.zeros(size)
        times[x != 0] = 1 / x[x != 0]
        times *= time / times.max()  # Extended through simulation time.
        # times = torch.ceil(times).long()
        # Create spike times tensor.
        spikes = torch.zeros(time, size).byte()

        for i in range(size):

            max_value, max_index = torch.max(times, dim=0, keepdim=True)
            # max_index 包含了最大值的索引，现在我们可以使用这个索引来更新spikes张量,将 spikes 中对应最大值索引的位置设为 1
            spikes[max_index, i] = 1
            times[max_index] = 0

        return spikes.reshape(time, *shape)




