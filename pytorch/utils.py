#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time: 2021-06-29 3:35 下午
    
Author: huayang
    
Subject: Utils for Pytorch

"""
import os
import doctest

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from torch import Tensor

from my.python.utils import get_logger
from my.pytorch.backend.tensor_op import l2_normalize

logger = get_logger(__name__)


def cosine_similarity_dense(x1, x2):
    """ cosine 距离（全连接）
        即 x1 中每个向量与 x2 中每个向量计算 cosine 距离，相当于计算一个 attention 矩阵

        等价于 `F.cosine_similarity(x1.unsqueeze(1), x1.unsqueeze(0), dim=-1)`
    Args:
        x1: [B1, N]
        x2: [B2, N]

    Returns:
        [B1, B2] matrix
    """
    assert x1.ndim == x2.ndim == 2

    x1_normalized = l2_normalize(x1, dim=-1)  # [B1, N]
    x2_normalized_T = l2_normalize(x2, dim=-1).T  # [N, B2]
    return torch.matmul(x1_normalized, x2_normalized_T)  # [B1, B2]


def create_mask_3d(q_tensor: Tensor, v_mask: Tensor, dtype=torch.float):
    """ Create 3D attention mask from a 2D tensor mask.

    Args:
      q_tensor: 2D or 3D Tensor of shape [B, Q, ...].
      v_mask: int32 Tensor of shape [B, V].
      dtype:

    Returns:
        float Tensor of shape [B, Q, V].

    References:
        [google-research/bert](https://github.com/google-research/bert)
    """
    B = q_tensor.shape[0]  # B
    Q = q_tensor.shape[1]  # Q

    v_mask = v_mask.unsqueeze(1)  # [B, V] -> [B, 1, V]
    mask = torch.ones([B, Q, 1]) * v_mask  # [B, Q, V]
    return mask.to(dtype)


def get_state_dict(weights_path, from_tf=False):
    """ 加载预训练权重字典 {weight_name: tensor} """
    if from_tf:
        state_dict = load_state_dict_tf(weights_path)
    else:
        state_dict = load_state_dict_pt(weights_path)
        _update_state_dict_keys(state_dict)

    return state_dict


def get_version():
    """"""
    return torch.__version__


def load_state_dict_tf(weights_path):
    """"""
    import tensorflow as tf
    _loader = lambda name: tf.train.load_variable(weights_path, name)

    if os.path.isdir(weights_path):  # 如果是目录
        # 找出目录下的 xxx.ckpt.index 文件
        file_ls = os.listdir(weights_path)
        file_name = [f for f in file_ls if f.endswith('.index')][0]
        weights_path = os.path.join(weights_path, file_name)

    weights_path = weights_path[:-6] if weights_path.endswith('.index') else weights_path
    weights_pretrained = OrderedDict()
    for n, _ in tf.train.list_variables(weights_path):
        array = _loader(n)
        if n.endswith('kernel'):
            array = np.transpose(array)  # transpose(tf[in, out]) -> pt[out, in]
        weights_pretrained[n] = torch.as_tensor(array)

    return weights_pretrained


def load_state_dict_pt(weights_path, map_location='cpu'):
    """"""
    return torch.load(weights_path, map_location=map_location)


def load_weights_partly(model: nn.Module, weights_dict, name_mapping=None):
    """ 部分权重加载

    Args:
        model: 待加载模型
        weights_dict: {name: tensor} 字典
        name_mapping: {name: name_pre} 字典，默认为 None；
            当 weights_dict 雨模型中权重名称不匹配时，可以通过 name_mapping 再映射一次
    """
    if name_mapping:
        for name, name_pre in name_mapping.items():
            if name_pre in weights_dict:
                weights_dict[name] = weights_dict.pop(name_pre)  # 替换新名称

    load_keys = set()  # 记录顺利加载的 key
    state_dict_new = OrderedDict()  # 新 state_dict，不直接修改原 state_dict
    state_dict = model.state_dict()
    for name, tensor in state_dict.items():
        if name not in weights_dict:
            state_dict_new[name] = tensor
        else:
            _assert_shape(weights_dict[name], tensor)  # noqa

            state_dict_new[name] = weights_dict[name]
            load_keys.add(name)

    missed_keys = sorted(set(state_dict.keys()) - load_keys)  # 未更新的权重
    unused_keys = sorted(set(weights_dict.keys()) - load_keys)  # 未使用的权重
    logger.info(f'Missed keys({len(missed_keys)}): {missed_keys}')
    logger.info(f'Unused keys({len(unused_keys)}): {unused_keys}')

    model.load_state_dict(state_dict_new)  # reload
    model.eval()  # deactivate dropout
    return model


def log_softmax(x: Tensor, dim=-1):
    """"""
    x = softmax(x, dim=dim)  # [B, C]
    return torch.log(x)  # [B, C]


def sequence_masking(x: torch.Tensor,
                     mask: torch.Tensor,
                     axis=1, mode='add', inf=1e12):
    """序列 mask

    Args:
        x: 2D 或 2D 以上张量，必须包含 batch_size 和 seq_len 两个维度
        mask: 形如  (batch_size, seq_len) 的 0/1 矩阵
        axis: 需要 mask 的维度，即 seq_len 所在维度，默认为 1
        mode: 有 'mul' 和 'add' 两种：
            mul 会将 pad 部分置零，一般用于全连接层之前；
            add 会把 pad 部分减去一个大的常数，一般用于 softmax 之前。
        inf: 大的常数

    Returns:
        tensor with shape same as x

    Examples:
        mask = [B, L]
        示例 1：x.shape = [B, L, _],     则 axis=1 (默认)
        示例 2：x.shape = [B, _, L, _],  则 axis=2
        示例 3：x.shape = [B, _, _, L],  则 axis=-1
    """
    if mask is None:
        return x

    assert mask.ndim == 2, 'only for mask.ndim == 2'

    if axis < 0:
        axis = x.ndim + axis

    # 将 mask 的维度扩充到与 x 一致，以便进行广播
    # 示例：假设 x.shape = [B, _, L, _]
    # 则经过以下操作后，mask.shape = [B, 1, L, 1]，相当于 mask = mask[:, None, :, None]
    for _ in range(axis - 1):
        mask = mask.unsqueeze(1)
    for _ in range(x.ndim - axis - 1):
        mask = mask.unsqueeze(-1)

    if mode == 'mul':
        return x * mask
    elif mode == 'add':
        return x - (1 - mask) * inf
    else:
        raise ValueError('`mode` must be one of %s' % {'add', 'mul'})


def softmax(x: Tensor, dim=-1):
    """"""
    x_exp = torch.exp(x)  # [B, C]
    x_exp_sum = torch.sum(x_exp, dim=dim, keepdim=True)  # [B, 1]
    return x_exp / x_exp_sum  # [B, C]


def _assert_shape(tensor1, tensor2):
    """"""
    t1_shape = list(tensor1.shape)
    t2_shape = list(tensor2.shape)
    assert t1_shape == t2_shape, f'shape mismatching: {t1_shape} vs {t2_shape}'
    return True


def _update_state_dict_keys(state_dict):
    """"""
    # 特殊处理：一些 pt 权重参数中 LN 层的参数名依然为 gamma 和 beta（官方实现应该为 weight 和 bias）
    #   推测应该是使用了自定义 LN 层，所以参数名称不同，这里需要特殊处理一下
    tmp_keys = []
    for key in state_dict.keys():
        new_key = None
        if "gamma" in key:
            new_key = key.replace("gamma", "weight")
        if "beta" in key:
            new_key = key.replace("beta", "bias")
        if new_key:
            tmp_keys.append((key, new_key))

    for old_key, new_key in tmp_keys:
        state_dict[new_key] = state_dict.pop(old_key)

    return state_dict


def _test():
    """"""
    doctest.testmod()

    def _test_softmax():
        """"""
        x = torch.randn(5, 6)
        assert torch.allclose(softmax(x), F.softmax(x, dim=-1), atol=1e-5)
        assert torch.allclose(log_softmax(x), F.log_softmax(x, dim=-1), atol=1e-5)

    _test_softmax()


if __name__ == '__main__':
    """"""
    _test()
