#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time: 2021-10-13 4:17 下午

Author: huayang

Subject:

"""
import os
import sys
import json
import doctest

from typing import *
from collections import defaultdict
from typing import Dict

import numpy as np
import torch

from torch import Tensor
from torch.utils.data import Dataset, DataLoader

__all__ = [
    'DictTensorDataset'
]


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
        return list(self.tensors_dict.values())[0].shape[0]


def _test():
    """"""
    doctest.testmod()


if __name__ == '__main__':
    """"""
    _test()
