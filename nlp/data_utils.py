#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time:
    2021-08-23 7:28 下午

Author:
    huayang

Subject:

"""
import doctest
import warnings

import numpy as np


def safe_indexing(x, indices=None):
    """
    Return items or rows from X using indices.

    Args:
        x:
        indices:

    References:
        sklearn.utils.safe_indexing
    """
    if indices is None:
        return x

    if hasattr(x, "shape"):  # for numpy
        try:
            return x.take(indices, axis=0)  # faster
        except:
            return x[indices]
    elif hasattr(x, "iloc"):  # for pandas
        indices = np.asarray(indices)
        indices = indices if indices.flags.writeable else indices.copy()
        try:
            return x.iloc[indices]
        except:
            return x.copy().iloc[indices]
    else:  # for python
        return [x[idx] for idx in indices]


def split(*arrays, split_size=0.2, random_seed=1, shuffle=True):
    """
    将数据按比例切分

    Args:
        *arrays:
        split_size: 切分比例，采用向上取整：ceil(6*0.3) = 2
        random_seed: 随机数种子
        shuffle: 是否打乱

    Examples:
        >>> data = [[0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7]]
        >>> xt, xv = split(*data, split_size=0.3, shuffle=False)
        >>> xt
        [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]
        >>> xv
        [[5, 6, 7], [5, 6, 7], [5, 6, 7]]
        
    Returns:
        x_train, x_val =  split(x)
        (a_train, b_train, c_train), (a_val, b_train, c_train) = split(a, b, c)
    """

    # assert
    lens = [len(x) for x in arrays]
    if len(set(lens)) > 1:
        raise ValueError('The length of each array must be same, but %r.' % lens)

    n_sample = lens[0]

    if shuffle:
        rs = np.random.RandomState(random_seed)
        idx = rs.permutation(n_sample)
    else:
        idx = np.arange(n_sample)

    n_val = int(np.ceil(split_size * n_sample))
    arr_train = [safe_indexing(x, idx[:-n_val]) for x in arrays]
    arr_val = [safe_indexing(x, idx[-n_val:]) for x in arrays]

    if len(arrays) == 1:
        return arr_train[0], arr_val[0]

    return arr_train, arr_val


def simple_split(*arrays, val_size=0.25, random_seed=1, shuffle=True):
    """
    将数据按比例切分

    Args:
        *arrays:
        val_size: 切分比例
        random_seed: 随机数种子
        shuffle: 是否打乱

    Examples:
        (a_train, b_train, c_train), (a_val, b_train, c_train) = split(a, b, c)
    """
    warnings.warn(f"Recommend using `{split.__name__}` instead", DeprecationWarning)

    # assert
    assert all(len(arrays[0]) == len(arr) for arr in arrays), "Size mismatch between tensors"

    if shuffle:
        rs = np.random.RandomState(random_seed)

        tmp = list(zip(*arrays))
        rs.shuffle(tmp)
        arrays = list(zip(*tmp))

    n_sample = len(arrays[0])
    n_val = int(np.ceil(val_size * n_sample))
    arr_val = [x[-n_val:] for x in arrays]
    arr_train = [x[:-n_val] for x in arrays]

    return arr_train, arr_val


def unzip(data):
    """
    unzip([[1,2,3], [1,2,3], ..]) -> [[1,1,..], [2,2,..], [3,3,..]]
    """
    try:
        out_type = type(data)
        in_type = type(data[0])
        return out_type([in_type(it) for it in zip(*data)])  # 尝试还原类型，不一定都能成功
    except:
        return zip(*data)


def _test():
    """"""
    doctest.testmod()

    def _test_split():
        """"""
        data = [[i for i in range(10)]] * 3
        data_a, data_b = simple_split(*data, shuffle=False)
        print(data_a)
        print(data_b)

    _test_split()

    def _test_unzip():
        """"""
        from sortedcollections import SegmentList
        data = SegmentList([[1, 2, 3], (1, 2, 3)])
        tmp = unzip(data)
        print(tmp)
        # for it in tmp:
        #     print(it)

    _test_unzip()


if __name__ == '__main__':
    """"""
    _test()
