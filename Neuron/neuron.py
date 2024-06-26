import functools
from abc import abstractmethod
from typing import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import logging
import surrogate, base


class BaseNode(base.MemoryModule):
    def __init__(self, v_threshold: float = 1., v_reset: float = 0.,
                 surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False,
                 step_mode='s'):
        """
        :param v_threshold: 神经元的阈值电压
        :param v_reset: 神经元的重置电压。如果不为 ``None``，当神经元释放脉冲后，电压会被重置为 ``v_reset``；
            如果设置为 ``None``，当神经元释放脉冲后，电压会被减去 ``v_threshold``
        :param surrogate_function: 反向传播时用来计算脉冲函数梯度的替代函数
        可微分SNN神经元的基类神经元。

        """
        assert isinstance(v_reset, float) or v_reset is None
        assert isinstance(v_threshold, float)
        assert isinstance(detach_reset, bool)
        super().__init__()

        #base中的状态注册器
        if v_reset is None:
            self.register_memory('v', 0.)
        else:
            self.register_memory('v', v_reset)

        self.v_threshold = v_threshold
        self.v_reset = v_reset

        self.detach_reset = detach_reset
        self.surrogate_function = surrogate_function

        self.step_mode = step_mode
        # self.backend = backend
        #
        # self.store_v_seq = store_v_seq

        # used in lava_exchange
        # self.lava_s_cale = 1 << 6

    @property
    def store_v_seq(self):
        return self._store_v_seq

    @store_v_seq.setter
    def store_v_seq(self, value: bool):
        self._store_v_seq = value
        if value:
            if not hasattr(self, 'v_seq'):
                self.register_memory('v_seq', None)

    @abstractmethod
    def neuronal_charge(self, x: torch.Tensor):
        """
        定义神经元的充电差分方程。子类必须实现这个函数。
        """
        raise NotImplementedError

    def neuronal_fire(self):
        """
        根据当前神经元的电压、阈值，计算输出脉冲。
        """
        return self.surrogate_function(self.v - self.v_threshold)

    @staticmethod
    @torch.jit.script
    def jit_hard_reset(v: torch.Tensor, spike: torch.Tensor, v_reset: float):
        v = (1. - spike) * v + spike * v_reset
        return v

    @staticmethod
    @torch.jit.script
    def jit_soft_reset(v: torch.Tensor, spike: torch.Tensor, v_threshold: float):
        v = v - spike * v_threshold
        return v

    def neuronal_reset(self, spike):
        """
        根据当前神经元释放的脉冲，对膜电位进行重置。
        """
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike

        if self.v_reset is None:
            # soft reset
            self.v = self.jit_soft_reset(self.v, spike_d, self.v_threshold)

        else:
            # hard reset
            self.v = self.jit_hard_reset(self.v, spike_d, self.v_reset)

    def extra_repr(self):
        return f'v_threshold={self.v_threshold}, v_reset={self.v_reset}, detach_reset={self.detach_reset}, step_mode={self.step_mode}, backend={self.backend}'

    def single_step_forward(self, x: torch.Tensor):
        """
        :param x: 输入到神经元的电压增量
        :return: 神经元的输出脉冲
        按照充电、放电、重置的顺序进行前向传播。
        """
        self.v_float_to_tensor(x)
        self.neuronal_charge(x)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        return spike

    # def multi_step_forward(self, x_seq: torch.Tensor):
    #     T = x_seq.shape[0]
    #     y_seq = []
    #     if self.store_v_seq:
    #         v_seq = []
    #     for t in range(T):
    #         y = self.single_step_forward(x_seq[t])
    #         y_seq.append(y)
    #         if self.store_v_seq:
    #             v_seq.append(self.v)
    #
    #     if self.store_v_seq:
    #         self.v_seq = torch.stack(v_seq)
    #
    #     return torch.stack(y_seq)

    def v_float_to_tensor(self, x: torch.Tensor):
        if isinstance(self.v, float):
            v_init = self.v
            self.v = torch.full_like(x.data, v_init)


class IFNode(BaseNode):
    def __init__(self, v_threshold: float = 1., v_reset: float = 0.,
                 surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False, step_mode='s',):
        """
        :param v_threshold: 神经元的阈值电压
        :param v_reset: 神经元的重置电压。如果不为 ``None``，当神经元释放脉冲后，电压会被重置为 ``v_reset``；
            如果设置为 ``None``，当神经元释放脉冲后，电压会被减去 ``v_threshold``
        :param surrogate_function: 反向传播时用来计算脉冲函数梯度的替代函数
        Integrate-and-Fire 神经元模型，可以看作理想积分器，无输入时电压保持恒定，不会像LIF神经元那样衰减。其阈下神经动力学方程为：
        .. math:H[t] = V[t-1] + X[t]
        """
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset, step_mode)


    def neuronal_charge(self, x: torch.Tensor):
        self.v = self.v + x

    @staticmethod
    @torch.jit.script
    def jit_eval_single_step_forward_hard_reset(x: torch.Tensor, v: torch.Tensor, v_threshold: float, v_reset: float):
        v = v + x
        spike = (v >= v_threshold).to(x)
        v = v_reset * spike + (1. - spike) * v
        return spike, v

    @staticmethod
    @torch.jit.script
    def jit_eval_single_step_forward_soft_reset(x: torch.Tensor, v: torch.Tensor, v_threshold: float):
        v = v + x
        spike = (v >= v_threshold).to(x)
        v = v - spike * v_threshold
        return spike, v
    #
    # @staticmethod
    # @torch.jit.script
    # def jit_eval_multi_step_forward_hard_reset(x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float,
    #                                            v_reset: float):
    #     spike_seq = torch.zeros_like(x_seq)
    #     for t in range(x_seq.shape[0]):
    #         v = v + x_seq[t]
    #         spike = (v >= v_threshold).to(x_seq)
    #         v = v_reset * spike + (1. - spike) * v
    #         spike_seq[t] = spike
    #     return spike_seq, v
    #
    # @staticmethod
    # @torch.jit.script
    # def jit_eval_multi_step_forward_hard_reset_with_v_seq(x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float,
    #                                                       v_reset: float):
    #     spike_seq = torch.zeros_like(x_seq)
    #     v_seq = torch.zeros_like(x_seq)
    #     for t in range(x_seq.shape[0]):
    #         v = v + x_seq[t]
    #         spike = (v >= v_threshold).to(x_seq)
    #         v = v_reset * spike + (1. - spike) * v
    #         spike_seq[t] = spike
    #         v_seq[t] = v
    #     return spike_seq, v, v_seq

    # @staticmethod
    # @torch.jit.script
    # def jit_eval_multi_step_forward_soft_reset(x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float):
    #     spike_seq = torch.zeros_like(x_seq)
    #     for t in range(x_seq.shape[0]):
    #         v = v + x_seq[t]
    #         spike = (v >= v_threshold).to(x_seq)
    #         v = v - spike * v_threshold
    #         spike_seq[t] = spike
    #     return spike_seq, v
    #
    # @staticmethod
    # @torch.jit.script
    # def jit_eval_multi_step_forward_soft_reset_with_v_seq(x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float):
    #     spike_seq = torch.zeros_like(x_seq)
    #     v_seq = torch.zeros_like(x_seq)
    #     for t in range(x_seq.shape[0]):
    #         v = v + x_seq[t]
    #         spike = (v >= v_threshold).to(x_seq)
    #         v = v - spike * v_threshold
    #         spike_seq[t] = spike
    #         v_seq[t] = v
    #     return spike_seq, v, v_seq
    #
    # def multi_step_forward(self, x_seq: torch.Tensor):
    #     if self.training:
    #         if self.backend == 'torch':
    #             return super().multi_step_forward(x_seq)
    #         else:
    #             raise ValueError(self.backend)
    #
    #     else:
    #         self.v_float_to_tensor(x_seq[0])
    #         if self.v_reset is None:
    #             if self.store_v_seq:
    #                 spike_seq, self.v, self.v_seq = self.jit_eval_multi_step_forward_soft_reset_with_v_seq(x_seq,
    #                                                                                                        self.v,
    #                                                                                                        self.v_threshold)
    #             else:
    #                 spike_seq, self.v = self.jit_eval_multi_step_forward_soft_reset(x_seq, self.v, self.v_threshold)
    #         else:
    #             if self.store_v_seq:
    #                 spike_seq, self.v, self.v_seq = self.jit_eval_multi_step_forward_hard_reset_with_v_seq(x_seq,
    #                                                                                                        self.v,
    #                                                                                                        self.v_threshold,
    #                                                                                                        self.v_reset)
    #             else:
    #                 spike_seq, self.v = self.jit_eval_multi_step_forward_hard_reset(x_seq, self.v, self.v_threshold,
    #                                                                                 self.v_reset)
    #         return spike_seq

    def single_step_forward(self, x: torch.Tensor):
        self.v_float_to_tensor(x)
        if self.v_reset is None:
            spike, self.v = self.jit_eval_single_step_forward_soft_reset(x, self.v, self.v_threshold)
        else:
            spike, self.v = self.jit_eval_single_step_forward_hard_reset(x, self.v, self.v_threshold, self.v_reset)
        return spike


class LIFNode(BaseNode):
    def __init__(self, tau: float = 2., decay_input: bool = True, v_threshold: float = 1.,
                 v_reset: float = 0., surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False, step_mode='s', backend='torch', store_v_seq: bool = False):
        """
        :param tau: 膜电位时间常数
        :param decay_input: 输入是否也会参与衰减
        :param v_threshold: 神经元的阈值电压
        :param v_reset: 神经元的重置电压。如果不为 ``None``，当神经元释放脉冲后，电压会被重置为 ``v_reset``；
            如果设置为 ``None``，当神经元释放脉冲后，电压会被减去 ``v_threshold``
        :param surrogate_function: 反向传播时用来计算脉冲函数梯度的替代函数
        Leaky Integrate-and-Fire 神经元模型，可以看作是带漏电的积分器。其阈下神经动力学方程为：

        若 ``decay_input == True``:
            .. math::
                H[t] = V[t-1] + \\frac{1}{\\tau}(X[t] - (V[t-1] - V_{reset}))
        若 ``decay_input == False``:

            .. math::
                H[t] = V[t-1] - \\frac{1}{\\tau}(V[t-1] - V_{reset}) + X[t]
        """
        assert isinstance(tau, float) and tau > 1.

        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset, step_mode)

        self.tau = tau
        self.decay_input = decay_input


    def extra_repr(self):
        return super().extra_repr() + f', tau={self.tau}'

    def neuronal_charge(self, x: torch.Tensor):
        if self.decay_input:
            if self.v_reset is None or self.v_reset == 0.:
                self.v = self.neuronal_charge_decay_input_reset0(x, self.v, self.tau)
            else:
                self.v = self.neuronal_charge_decay_input(x, self.v, self.v_reset, self.tau)

        else:
            if self.v_reset is None or self.v_reset == 0.:
                self.v = self.neuronal_charge_no_decay_input_reset0(x, self.v, self.tau)
            else:
                self.v = self.neuronal_charge_no_decay_input(x, self.v, self.v_reset, self.tau)

    @staticmethod
    @torch.jit.script
    def neuronal_charge_decay_input_reset0(x: torch.Tensor, v: torch.Tensor, tau: float):
        v = v + (x - v) / tau
        return v

    @staticmethod
    @torch.jit.script
    def neuronal_charge_decay_input(x: torch.Tensor, v: torch.Tensor, v_reset: float, tau: float):
        v = v + (x - (v - v_reset)) / tau
        return v

    @staticmethod
    @torch.jit.script
    def neuronal_charge_no_decay_input_reset0(x: torch.Tensor, v: torch.Tensor, tau: float):
        v = v * (1. - 1. / tau) + x
        return v

    @staticmethod
    @torch.jit.script
    def neuronal_charge_no_decay_input(x: torch.Tensor, v: torch.Tensor, v_reset: float, tau: float):
        v = v - (v - v_reset) / tau + x
        return v

    @staticmethod
    @torch.jit.script
    def jit_eval_single_step_forward_hard_reset_decay_input(x: torch.Tensor, v: torch.Tensor, v_threshold: float,
                                                            v_reset: float, tau: float):
        v = v + (x - (v - v_reset)) / tau
        spike = (v >= v_threshold).to(x)
        v = v_reset * spike + (1. - spike) * v
        return spike, v

    @staticmethod
    @torch.jit.script
    def jit_eval_single_step_forward_hard_reset_no_decay_input(x: torch.Tensor, v: torch.Tensor, v_threshold: float,
                                                               v_reset: float, tau: float):
        v = v - (v - v_reset) / tau + x
        spike = (v >= v_threshold).to(x)
        v = v_reset * spike + (1. - spike) * v
        return spike, v

    @staticmethod
    @torch.jit.script
    def jit_eval_single_step_forward_soft_reset_decay_input(x: torch.Tensor, v: torch.Tensor, v_threshold: float,
                                                            tau: float):
        v = v + (x - v) / tau
        spike = (v >= v_threshold).to(x)
        v = v - spike * v_threshold
        return spike, v

    @staticmethod
    @torch.jit.script
    def jit_eval_single_step_forward_soft_reset_no_decay_input(x: torch.Tensor, v: torch.Tensor, v_threshold: float,
                                                               tau: float):
        v = v * (1. - 1. / tau) + x
        spike = (v >= v_threshold).to(x)
        v = v - spike * v_threshold
        return spike, v
    #
    # @staticmethod
    # @torch.jit.script
    # def jit_eval_multi_step_forward_hard_reset_decay_input(x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float,
    #                                                        v_reset: float, tau: float):
    #     spike_seq = torch.zeros_like(x_seq)
    #     for t in range(x_seq.shape[0]):
    #         v = v + (x_seq[t] - (v - v_reset)) / tau
    #         spike = (v >= v_threshold).to(x_seq)
    #         v = v_reset * spike + (1. - spike) * v
    #         spike_seq[t] = spike
    #     return spike_seq, v
    #
    # @staticmethod
    # @torch.jit.script
    # def jit_eval_multi_step_forward_hard_reset_decay_input_with_v_seq(x_seq: torch.Tensor, v: torch.Tensor,
    #                                                                   v_threshold: float, v_reset: float, tau: float):
    #     spike_seq = torch.zeros_like(x_seq)
    #     v_seq = torch.zeros_like(x_seq)
    #     for t in range(x_seq.shape[0]):
    #         v = v + (x_seq[t] - (v - v_reset)) / tau
    #         spike = (v >= v_threshold).to(x_seq)
    #         v = v_reset * spike + (1. - spike) * v
    #         spike_seq[t] = spike
    #         v_seq[t] = v
    #     return spike_seq, v, v_seq
    #
    # @staticmethod
    # @torch.jit.script
    # def jit_eval_multi_step_forward_hard_reset_no_decay_input(x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float,
    #                                                           v_reset: float, tau: float):
    #     spike_seq = torch.zeros_like(x_seq)
    #     for t in range(x_seq.shape[0]):
    #         v = v - (v - v_reset) / tau + x_seq[t]
    #         spike = (v >= v_threshold).to(x_seq)
    #         v = v_reset * spike + (1. - spike) * v
    #         spike_seq[t] = spike
    #     return spike_seq, v
    #
    # @staticmethod
    # @torch.jit.script
    # def jit_eval_multi_step_forward_hard_reset_no_decay_input_with_v_seq(x_seq: torch.Tensor, v: torch.Tensor,
    #                                                                      v_threshold: float, v_reset: float,
    #                                                                      tau: float):
    #     spike_seq = torch.zeros_like(x_seq)
    #     v_seq = torch.zeros_like(x_seq)
    #     for t in range(x_seq.shape[0]):
    #         v = v - (v - v_reset) / tau + x_seq[t]
    #         spike = (v >= v_threshold).to(x_seq)
    #         v = v_reset * spike + (1. - spike) * v
    #         spike_seq[t] = spike
    #         v_seq[t] = v
    #     return spike_seq, v, v_seq
    #
    # @staticmethod
    # @torch.jit.script
    # def jit_eval_multi_step_forward_soft_reset_decay_input(x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float,
    #                                                        tau: float):
    #     spike_seq = torch.zeros_like(x_seq)
    #     for t in range(x_seq.shape[0]):
    #         v = v + (x_seq[t] - v) / tau
    #         spike = (v >= v_threshold).to(x_seq)
    #         v = v - spike * v_threshold
    #         spike_seq[t] = spike
    #     return spike_seq, v
    #
    # @staticmethod
    # @torch.jit.script
    # def jit_eval_multi_step_forward_soft_reset_decay_input_with_v_seq(x_seq: torch.Tensor, v: torch.Tensor,
    #                                                                   v_threshold: float, tau: float):
    #     spike_seq = torch.zeros_like(x_seq)
    #     v_seq = torch.zeros_like(x_seq)
    #     for t in range(x_seq.shape[0]):
    #         v = v + (x_seq[t] - v) / tau
    #         spike = (v >= v_threshold).to(x_seq)
    #         v = v - spike * v_threshold
    #         spike_seq[t] = spike
    #         v_seq[t] = v
    #     return spike_seq, v, v_seq
    #
    # @staticmethod
    # @torch.jit.script
    # def jit_eval_multi_step_forward_soft_reset_no_decay_input(x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float,
    #                                                           tau: float):
    #     spike_seq = torch.zeros_like(x_seq)
    #     for t in range(x_seq.shape[0]):
    #         v = v * (1. - 1. / tau) + x_seq[t]
    #         spike = (v >= v_threshold).to(x_seq)
    #         v = v - spike * v_threshold
    #         spike_seq[t] = spike
    #     return spike_seq, v
    #
    # @staticmethod
    # @torch.jit.script
    # def jit_eval_multi_step_forward_soft_reset_no_decay_input_with_v_seq(x_seq: torch.Tensor, v: torch.Tensor,
    #                                                                      v_threshold: float,
    #                                                                      tau: float):
    #     spike_seq = torch.zeros_like(x_seq)
    #     v_seq = torch.zeros_like(x_seq)
    #     for t in range(x_seq.shape[0]):
    #         v = v * (1. - 1. / tau) + x_seq[t]
    #         spike = (v >= v_threshold).to(x_seq)
    #         v = v - spike * v_threshold
    #         spike_seq[t] = spike
    #         v_seq[t] = v
    #     return spike_seq, v, v_seq

    def single_step_forward(self, x: torch.Tensor):
        self.v_float_to_tensor(x)
        if self.v_reset is None:
            if self.decay_input:
                spike, self.v = self.jit_eval_single_step_forward_soft_reset_decay_input(x, self.v,
                                                                                         self.v_threshold, self.tau)
            else:
                spike, self.v = self.jit_eval_single_step_forward_soft_reset_no_decay_input(x, self.v,
                                                                                            self.v_threshold,
                                                                                            self.tau)
        else:
            if self.decay_input:
                spike, self.v = self.jit_eval_single_step_forward_hard_reset_decay_input(x, self.v,
                                                                                         self.v_threshold,
                                                                                         self.v_reset, self.tau)
            else:
                spike, self.v = self.jit_eval_single_step_forward_hard_reset_no_decay_input(x, self.v,
                                                                                            self.v_threshold,
                                                                                            self.v_reset,
                                                                                            self.tau)
        return spike

    # def multi_step_forward(self, x_seq: torch.Tensor):
    #     if self.training:
    #         if self.backend == 'torch':
    #             return super().multi_step_forward(x_seq)
    #
    #         else:
    #             raise ValueError(self.backend)
    #
    #     else:
    #         self.v_float_to_tensor(x_seq[0])
    #         if self.v_reset is None:
    #             if self.decay_input:
    #                 if self.store_v_seq:
    #                     spike_seq, self.v, self.v_seq = self.jit_eval_multi_step_forward_soft_reset_decay_input_with_v_seq(
    #                         x_seq, self.v, self.v_threshold, self.tau)
    #                 else:
    #                     spike_seq, self.v = self.jit_eval_multi_step_forward_soft_reset_decay_input(x_seq, self.v,
    #                                                                                                 self.v_threshold,
    #                                                                                                 self.tau)
    #             else:
    #                 if self.store_v_seq:
    #                     spike_seq, self.v, self.v_seq = self.jit_eval_multi_step_forward_soft_reset_no_decay_input_with_v_seq(
    #                         x_seq, self.v, self.v_threshold, self.tau)
    #                 else:
    #                     spike_seq, self.v = self.jit_eval_multi_step_forward_soft_reset_no_decay_input(x_seq, self.v,
    #                                                                                                    self.v_threshold,
    #                                                                                                    self.tau)
    #         else:
    #             if self.decay_input:
    #                 if self.store_v_seq:
    #                     spike_seq, self.v, self.v_seq = self.jit_eval_multi_step_forward_hard_reset_decay_input_with_v_seq(
    #                         x_seq, self.v, self.v_threshold, self.v_reset, self.tau)
    #                 else:
    #                     spike_seq, self.v = self.jit_eval_multi_step_forward_hard_reset_decay_input(x_seq, self.v,
    #                                                                                                 self.v_threshold,
    #                                                                                                 self.v_reset,
    #                                                                                                 self.tau)
    #             else:
    #                 if self.store_v_seq:
    #                     spike_seq, self.v, self.v_seq = self.jit_eval_multi_step_forward_hard_reset_no_decay_input_with_v_seq(
    #                         x_seq, self.v, self.v_threshold, self.v_reset, self.tau)
    #                 else:
    #                     spike_seq, self.v = self.jit_eval_multi_step_forward_hard_reset_no_decay_input(x_seq, self.v,
    #                                                                                                    self.v_threshold,
    #                                                                                                    self.v_reset,
    #                                                                                                    self.tau)
    #
    #         return spike_seq


class SRMNode(BaseNode):

    def __init__(self, tau_m=10, tau_s=5, t_current=0.3, t_membrane=20, eta_reset=5, v_threshold: float = 1., v_reset: float = 0.,
                 surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False, step_mode='s', current_time: float = 1):
        """
        :param threshold: 阈值
        :param tau_m: tau_m通常是膜时间常数，表示膜电位的恢复速度。较小的τm值表示神经元的膜电位将更快地恢复到其静息电位
        :param tau_s: 表示突触后电流的持续时间。较小的τs值表示突触后电流将更快地衰减，反之亦然
        :param v_reset: 静息电位
        """
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset, step_mode)
        self.current_time = current_time
        self.t_membrane = t_membrane
        self.t_current = t_current
        self.eta_reset = eta_reset
        self.tau_m = tau_m
        self.tau_s = tau_s

    def neuronal_charge(self, x: torch.Tensor):
            self.v = self.v + x

    #其他写法（考虑）
    # def eta(self, s):
    #     r"""
    #     Evaluate the Eta function:
    #     .. math:: \eta (s) = - \eta_{0} * \exp(\frac{- s}{\tau_{recov}})
    #     :param s: Time s
    #     :return: Function eta(s) at time s
    #     :return type: Float or Vector of Floats
    #     """
    #
    #     return - self.eta_reset*np.exp(-s/self.t_membrane)
    #
    # @functools.lru_cache()
    # def eps(self, s):
    #     r"""
    #     Evaluate the Epsilon function:
    #     .. math:: \epsilon (s) =  \frac{1}{1 - \frac{\tau_s}{\tau_m}} (\exp(\frac{-s}{\tau_m}) - \exp(\frac{-s}{\tau_s}))
    #     Returns a single Float Value if the time constants (current, membrane) are the same for each neuron.
    #     Returns a Float Vector with eps(s) for each neuron, if the time constants are different for each neuron.
    #     :param s: Time s
    #     :return: Function eps(s) at time s
    #     :rtype: Float or Vector of Floats
    #     """
    #     return (1/(1-self.current_time/self.t_membrane))*(np.exp(-s/self.t_membrane) - np.exp(-s/self.current_time))

    # def jit_eval_single_step_forward_soft_reset(self, current_time,last_spike_time, x: torch.Tensor, v: torch.Tensor, v_threshold: float):
    #     v += self.eps(current_time - self.last_spike_time) + self.eta(current_time - self.last_spike_time)
    #     spike = (v >= v_threshold).to(x)
    #     # 获取激发神经元的索引
    #     spike_indices = spike.nonzero(as_tuple=True)
    #
    #     # 计算时间差
    #     if len(spike_indices[0]) > 0:
    #         last_spike_time[spike_indices] = current_time
    #     v = v - spike * v_threshold
    #     return spike, v

    """在下面代码中，使用了近似的形式，忽略了膜电压的漏电流部分中的exp() 部分，直接在每个时间步长内按比例更新膜电压"""
    def jit_eval_single_step_forward_hard_reset(self, current_time, last_spike_time, x: torch.Tensor, v: torch.Tensor, v_threshold: float, v_reset: float):
        v += (self.v_reset - v) / self.tau_m
        v += x / self.tau_s
        spike = (v >= v_threshold).to(x)
        v = v_reset * spike + (1. - spike) * v
        return spike, v

    def jit_eval_single_step_forward_soft_reset(self, current_time,last_spike_time, x: torch.Tensor, v: torch.Tensor, v_threshold: float):
        v += (self.v_reset - v) / self.tau_m
        v += x / self.tau_s
        spike = (v >= v_threshold).to(x)
        v = v - spike * v_threshold
        return spike, v


    #
    # @staticmethod
    # @torch.jit.script
    # def jit_eval_multi_step_forward_hard_reset(x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float,
    #                                            v_reset: float):
    #     spike_seq = torch.zeros_like(x_seq)
    #     for t in range(x_seq.shape[0]):
    #         v = v + x_seq[t]
    #         spike = (v >= v_threshold).to(x_seq)
    #         v = v_reset * spike + (1. - spike) * v
    #         spike_seq[t] = spike
    #     return spike_seq, v
    #
    # @staticmethod
    # @torch.jit.script
    # def jit_eval_multi_step_forward_hard_reset_with_v_seq(x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float,
    #                                                       v_reset: float):
    #     spike_seq = torch.zeros_like(x_seq)
    #     v_seq = torch.zeros_like(x_seq)
    #     for t in range(x_seq.shape[0]):
    #         v = v + x_seq[t]
    #         spike = (v >= v_threshold).to(x_seq)
    #         v = v_reset * spike + (1. - spike) * v
    #         spike_seq[t] = spike
    #         v_seq[t] = v
    #     return spike_seq, v, v_seq

    # @staticmethod
    # @torch.jit.script
    # def jit_eval_multi_step_forward_soft_reset(x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float):
    #     spike_seq = torch.zeros_like(x_seq)
    #     for t in range(x_seq.shape[0]):
    #         v = v + x_seq[t]
    #         spike = (v >= v_threshold).to(x_seq)
    #         v = v - spike * v_threshold
    #         spike_seq[t] = spike
    #     return spike_seq, v
    #
    # @staticmethod
    # @torch.jit.script
    # def jit_eval_multi_step_forward_soft_reset_with_v_seq(x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float):
    #     spike_seq = torch.zeros_like(x_seq)
    #     v_seq = torch.zeros_like(x_seq)
    #     for t in range(x_seq.shape[0]):
    #         v = v + x_seq[t]
    #         spike = (v >= v_threshold).to(x_seq)
    #         v = v - spike * v_threshold
    #         spike_seq[t] = spike
    #         v_seq[t] = v
    #     return spike_seq, v, v_seq
    #
    # def multi_step_forward(self, x_seq: torch.Tensor):
    #     if self.training:
    #         if self.backend == 'torch':
    #             return super().multi_step_forward(x_seq)
    #         else:
    #             raise ValueError(self.backend)
    #
    #     else:
    #         self.v_float_to_tensor(x_seq[0])
    #         if self.v_reset is None:
    #             if self.store_v_seq:
    #                 spike_seq, self.v, self.v_seq = self.jit_eval_multi_step_forward_soft_reset_with_v_seq(x_seq,
    #                                                                                                        self.v,
    #                                                                                                        self.v_threshold)
    #             else:
    #                 spike_seq, self.v = self.jit_eval_multi_step_forward_soft_reset(x_seq, self.v, self.v_threshold)
    #         else:
    #             if self.store_v_seq:
    #                 spike_seq, self.v, self.v_seq = self.jit_eval_multi_step_forward_hard_reset_with_v_seq(x_seq,
    #                                                                                                        self.v,
    #                                                                                                        self.v_threshold,
    #                                                                                                        self.v_reset)
    #             else:
    #                 spike_seq, self.v = self.jit_eval_multi_step_forward_hard_reset(x_seq, self.v, self.v_threshold,
    #                                                                                 self.v_reset)
    #         return spike_seq

    def single_step_forward(self, x: torch.Tensor):
        self.v_float_to_tensor(x)
        last_spike_time = torch.zeros(x.shape)
        if self.v_reset is None:
            spike, self.v = self.jit_eval_single_step_forward_soft_reset(self.current_time, last_spike_time, x, self.v, self.v_threshold)
        else:
            spike, self.v = self.jit_eval_single_step_forward_hard_reset(self.current_time, last_spike_time, x, self.v, self.v_threshold, self.v_reset)
        return spike
