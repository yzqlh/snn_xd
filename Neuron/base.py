import torch
import torch.nn as nn
import copy
import logging
from abc import abstractmethod

# try:
#     import cupy
# except BaseException as e:
#     logging.info(f'spikingjelly.activation_based.base: {e}')
#     cupy = None
#
# try:
#     import lava.lib.dl.slayer as slayer
# except BaseException as e:
#     slayer = None


def check_backend_library(backend: str):
    """
    :param backend: ``'torch'``, ``'cupy'`` 或 ``'lava'``
    :type backend: str

    检查某个后端的python库是否已经安装。若未安装则此函数会报错。
    """
    if backend == 'torch':
        return
    # elif backend == 'cupy':
    #     if cupy is None:
    #         raise ImportError('CuPy is not installed! You can install it from "https://github.com/cupy/cupy".')
    # elif backend == 'lava':
    #     if slayer is None:
    #         raise ImportError('Lava-DL is not installed! You can install it from ' \
    #                           '"https://github.com/lava-nc/lava-dl". ')
    else:
        pass


class StepModule:
    def supported_step_mode(self):
        """
        :return: 包含支持的后端的tuple
        :rtype: tuple[str]

        返回此模块支持的步进模式。
        """
        return ('s', 'm')

    @property
    def step_mode(self):
        """
        :return: 模块当前使用的步进模式
        :rtype: str
        """
        return self._step_mode

    @step_mode.setter
    def step_mode(self, value: str):
        """
        :param value: 步进模式
        :type value: str
        将本模块的步进模式设置为 ``value``
        """
        if value not in self.supported_step_mode():
            raise ValueError(f'step_mode can only be {self.supported_step_mode()}, but got "{value}"!')
        self._step_mode = value


class SingleModule(StepModule):
    """
    只支持单步的模块 (``step_mode == 's'``)。
    """
    def supported_step_mode(self):
        return ('s', )

# class MultiStepModule(StepModule):
#     """
#     只支持多步的模块 (``step_mode == 'm'``)。
#     """
#     def supported_step_mode(self):
#         return ('m', )


class MemoryModule(nn.Module, StepModule):
    def __init__(self):
        """
        ``MemoryModule`` 是所有有状态（记忆）模块的基类。
        """
        super().__init__()
        self._memories = {}
        self._memories_rv = {}
        self._backend = 'torch'
        self.step_mode = 's'

    @property
    def supported_backends(self):
        """
        返回支持的后端，默认情况下只有 `('torch', )`
        :return: 支持的后端
        :rtype: tuple[str]
        """
        return ('torch',)

    @property
    def backend(self):
        return self._backend

    @backend.setter
    def backend(self, value: str):
        if value not in self.supported_backends:
            raise NotImplementedError(f'{value} is not a supported backend of {self._get_name()}!')
        check_backend_library(value)
        self._backend = value

    @abstractmethod
    def single_step_forward(self, x: torch.Tensor, *args, **kwargs):
        """
        :param x: input tensor with ``shape = [N, *] ``
        :type x: torch.Tensor
        本模块的单步的前向传播函数

        """
        pass

    # def multi_step_forward(self, x_seq: torch.Tensor, *args, **kwargs):
    #     """
    #     :param x: input tensor with ``shape = [T, N, *] ``
    #     :type x: torch.Tensor
    #
    #     本模块的多步的前向传播函数，通过调用 ``T`` 次 ``single_step_forward(x[t], *args, **kwargs)`` 实现
    #
    #     """
    #
    #     T = x_seq.shape[0]
    #     y_seq = []
    #     for t in range(T):
    #         y = self.single_step_forward(x_seq[t], *args, **kwargs)
    #         y_seq.append(y.unsqueeze(0))
    #
    #     return torch.cat(y_seq, 0)

    def forward(self, *args, **kwargs):
        if self.step_mode == 's':
            return self.single_step_forward(*args, **kwargs)
        elif self.step_mode == 'm':
            return self.multi_step_forward(*args, **kwargs)
        else:
            raise ValueError(self.step_mode)

    def extra_repr(self):
        return f'step_mode={self.step_mode}, backend={self.backend}'

    def register_memory(self, name: str, value):
        """
        :param name: 变量的名字
        :type name: str
        :param value: 变量的值
        :type value: any

        将变量存入用于保存有状态变量（例如脉冲神经元的膜电位）的字典中。这个变量的重置值会被设置为 ``value``。每次调用 ``self.reset()``
        函数后， ``self.name`` 都会被重置为 ``value``。

        """
        assert not hasattr(self, name), f'{name} has been set as a member variable!'
        self._memories[name] = value
        self.set_reset_value(name, value)

    def reset(self):
        """
        重置所有有状态变量为默认值。
        """
        for key in self._memories.keys():
            self._memories[key] = copy.deepcopy(self._memories_rv[key])

    def set_reset_value(self, name: str, value):
        self._memories_rv[name] = copy.deepcopy(value)

    def __getattr__(self, name: str):
        if '_memories' in self.__dict__:
            memories = self.__dict__['_memories']
            if name in memories:
                return memories[name]

        return super().__getattr__(name)

    def __setattr__(self, name: str, value) -> None:
        _memories = self.__dict__.get('_memories')
        if _memories is not None and name in _memories:
            _memories[name] = value
        else:
            super().__setattr__(name, value)

    def __delattr__(self, name):
        if name in self._memories:
            del self._memories[name]
            del self._memories_rv[name]
        else:
            return super().__delattr__(name)

    def __dir__(self):
        module_attrs = dir(self.__class__)
        attrs = list(self.__dict__.keys())
        parameters = list(self._parameters.keys())
        modules = list(self._modules.keys())
        buffers = list(self._buffers.keys())
        memories = list(self._memories.keys())
        keys = module_attrs + attrs + parameters + modules + buffers + memories

        # Eliminate attrs that are not legal Python variable names
        keys = [key for key in keys if not key[0].isdigit()]

        return sorted(keys)

    def memories(self):
        """
        :return: 返回一个所有状态变量的迭代器
        :rtype: Iterator
        """
        for name, value in self._memories.items():
            yield value

    def named_memories(self):
        """
        :return: 返回一个所有状态变量及其名称的迭代器
        :rtype: Iterator
        """

        for name, value in self._memories.items():
            yield name, value

    def _apply(self, fn):
        for key, value in self._memories.items():
            if isinstance(value, torch.Tensor):
                self._memories[key] = fn(value)
        # do not apply on default values
        # for key, value in self._memories_rv.items():
        #     if isinstance(value, torch.Tensor):
        #         self._memories_rv[key] = fn(value)
        return super()._apply(fn)

    def _replicate_for_data_parallel(self):
        replica = super()._replicate_for_data_parallel()
        replica._memories = self._memories.copy()
        return replica