#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time: 2021-10-13 4:21 下午

Author: huayang

Subject:

"""
import os
import sys
import json
import doctest

from typing import *
from collections import defaultdict
from typing import Iterable

import torch
from torch.utils.data import DataLoader, TensorDataset

from my.pytorch.data_utils.DictTensorDataset import DictTensorDataset
from my.pytorch.train.config import default_device

__all__ = [
    'ToyDataLoader'
]


class ToyDataLoader(DataLoader):
    """@Pytorch Utils
    一个简单的 DataLoader

    简化中间创建 Dataset 的过程，直接从数据（tensor/list/ndarray）创建 DataLoader

    Examples:
        >>> x = y = torch.as_tensor([1,2,3,4,5])
        >>> # 返回 tuple
        >>> dl = ToyDataLoader([x, y], batch_size=3, shuffle=False)
        >>> for batch in dl:
        ...     print(batch)
        [tensor([1, 2, 3]), tensor([1, 2, 3])]
        [tensor([4, 5]), tensor([4, 5])]
        >>> # 返回 dict
        >>> dl = ToyDataLoader({'x': x, 'y': y}, batch_size=3, shuffle=False)
        >>> for batch in dl:
        ...     print(batch)
        {'x': tensor([1, 2, 3]), 'y': tensor([1, 2, 3])}
        {'x': tensor([4, 5]), 'y': tensor([4, 5])}
    """

    def __init__(self, dataset: Iterable,
                 batch_size=16, shuffle=True, device=None, **kwargs):
        """"""
        if device is None:
            device = default_device()

        if isinstance(dataset, dict):
            dataset = {name: torch.as_tensor(tensor).to(device) for name, tensor in dataset.items()}
            dataset = DictTensorDataset(**dataset)
        else:
            dataset = [torch.as_tensor(tensor).to(device) for tensor in list(dataset)]
            dataset = TensorDataset(*dataset)

        # sampler = RandomSampler(dataset) if shuffle else None
        super(ToyDataLoader, self).__init__(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)


def _test():
    """"""
    doctest.testmod()


if __name__ == '__main__':
    """"""
    _test()
