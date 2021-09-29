#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time:
    2021-09-13 3:50 下午

Author:
    huayang

Subject:

"""
import os
import doctest

from argparse import Namespace
from typing import Union, Dict

import torch

from my.python import Config, get_time_string

DEFAULT_NO_DECAY_PARAMS = ('bias', 'LayerNorm.weight', 'layer_norm.weight', 'ln.weight')
DEFAULT_SAVE_DIR = os.path.join(os.environ['HOME'], 'out/models')

ARGS_TYPE = Union[Dict, Namespace]

# ** 不建议直接使用 _XXX 常量，而是调用相应方法 **
_DEFAULT_DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'
""" 对应方法 default_device() """


def default_device() -> str:
    """
    Examples:
        >>> assert default_device() == 'cuda' if torch.cuda.is_available() else 'cpu'

        # 通过以下方式全局修改 _DEFAULT_DEVICE
        >>> from my.pytorch.pipeline import config
        >>> from my.pytorch.pipeline.config import default_device  # doctest 时必须加这句，否则 setter 并不会生效
        >>> setattr(config, '_DEFAULT_DEVICE', 'xxx')
        >>> default_device()
        'xxx'
        >>> config._DEFAULT_DEVICE = 'cp'
        >>> default_device()
        'cp'

    """
    # return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return _DEFAULT_DEVICE


def set_device_cpu():
    """
    Examples:
        >>> set_device_cpu()
        >>> default_device()
        'cpu'

    """
    set_device('cpu')


def set_device(device: str):
    """
    Examples:
        >>> set_device('aaa')
        >>> default_device()
        'aaa'

    """
    global _DEFAULT_DEVICE
    _DEFAULT_DEVICE = device


_MAX_GPU_BATCH_SIZE = 16
""" 对应方法 max_gpu_batch_size() """


def max_gpu_batch_size():
    """"""
    return _MAX_GPU_BATCH_SIZE


def set_gpu_batch_size(gpu_batch_size: int):
    """"""
    global _MAX_GPU_BATCH_SIZE
    _MAX_GPU_BATCH_SIZE = gpu_batch_size


def get_model_save_dir():
    return os.path.join(DEFAULT_SAVE_DIR, f'model-{get_time_string(fmt="%Y%m%d%H%M%S")}')


class TrainConfig(Config):  # 父类基于 dict
    """"""

    def __init__(self, **config_items):
        """"""
        # data
        self.num_gpu: int = 1
        self.batch_size: int = 32
        self.shuffle = True
        self.val_percent = 0.2  # 从训练集中划分验证集的比例（如果没有提供验证集）
        self.random_seed = 1

        # device
        # self.device = DEFAULT_DEVICE  # 保险起见，通过引用复制，以防 config 没有全局应用
        self.device = default_device()

        # train
        self.learning_rate: float = 5e-5
        self.weight_decay: float = 0.01
        self.gpu_batch_size: int = max_gpu_batch_size()
        if self.batch_size > self.gpu_batch_size:
            self.num_gradient_accumulation: int = self.batch_size // self.gpu_batch_size  # 梯度累计，模拟更大的 batch_size
        else:
            self.num_gradient_accumulation = 1
        self.num_train_epochs: int = 3
        self.num_train_steps: int = -1
        self.num_warmup_steps: int = 0
        self.global_step: list = [0]  # int 类型为不可变类型，故使用列表引用类型
        self.optimizer_name = 'AdamW'
        self.no_decay_params = DEFAULT_NO_DECAY_PARAMS

        # save
        self.save_dir: str = get_model_save_dir()
        self.save_old_format: bool = True
        self.save_state_dict: bool = True

        super(TrainConfig, self).__init__(**config_items)


DEFAULT_ARGS = TrainConfig()


def _test():
    """"""
    doctest.testmod()

    t = TrainConfig()
    print(t)


if __name__ == '__main__':
    """"""
    _test()
