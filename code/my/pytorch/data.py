#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time: 2021-10-14 3:54 下午

Author: huayang

Subject:

"""
import os
import sys
import json
import doctest

from typing import *
from collections import defaultdict
from typing import Iterable, Dict

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset, Dataset

from my.pytorch.config import default_device

__all__ = [
    'ToyDataLoader',
    'DictTensorDataset'
]


def _test():
    """"""
    doctest.testmod()


if __name__ == '__main__':
    """"""
    _test()


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


class DictTensorDataset(Dataset[Dict[str, Tensor]]):
    """@Pytorch Utils
    字典形式的 Dataset

    使用本类生成 DataLoader 时，可以返回 dict 类型的 batch

    Examples:
        >>> x = y = torch.as_tensor([1,2,3,4,5])
        >>> ds = DictTensorDataset(x=x, y=y)
        >>> len(ds)
        5
        >>> dl = DataLoader(ds, batch_size=3)
        >>> for batch in dl: print(batch)
        {'x': tensor([1, 2, 3]), 'y': tensor([1, 2, 3])}
        {'x': tensor([4, 5]), 'y': tensor([4, 5])}
        >>> len(dl)
        2

    References:
        - torch.utils.data.TensorDataset
        - huggingface/datasets.arrow_dataset.Dataset
    """

    tensors_dict: Dict[str, Tensor]

    def __init__(self, **tensors_dict: Tensor) -> None:
        """
        Args:
            **tensors_dict:
        """
        assert len(np.unique([tensor.shape[0] for tensor in tensors_dict.values()])) == 1, \
            "Size mismatch between tensors"
        self.tensors_dict = tensors_dict

    def __getitem__(self, index) -> Dict[str, Tensor]:
        """"""
        return {name: tensor[index] for name, tensor in self.tensors_dict.items()}

    def __len__(self):
        return next(iter(self.tensors_dict.values())).shape[0]
