#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time: 2021-10-13 11:19 上午

Author: huayang

Subject:

"""
import os
import sys
import json
import doctest

from typing import *
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa

from my.pytorch.loss.BaseLoss import BaseLoss, _EPSILON

from my.pytorch.loss.loss_fn import _EPSILON

__all__ = [
    'CrossEntropyLoss'
]


def negative_likelihood_loss(logits, onehot_labels):
    """
    负似然损失，等价于 `F.nll_loss()`

    Examples:
        >>> _logits = torch.randn(5, 5).clamp(min=_EPSILON)  # 负对数似然的输入需要值大于 0
        >>> _labels = torch.arange(5)
        >>> _onehot_labels = F.one_hot(_labels)
        >>> # 与官方结果比较
        >>> my_ret = negative_likelihood_loss(_logits, _onehot_labels)
        >>> official_ret = F.nll_loss(_logits, _labels, reduction='none')
        >>> assert torch.allclose(my_ret, official_ret, atol=1e-5)

    Args:
        logits:
        onehot_labels:
    """
    return -(onehot_labels * logits).sum(-1)


def negative_log_likelihood_loss(logits, onehot_labels):
    """
    负对数似然损失，相比 `negative_likelihood_loss`，在计算损失之前，先对 `logits` 计算一次 `log`

        negative_likelihood_loss(log(logits)) <=> negative_log_likelihood_loss(logits)

    说明：计算 log 需确保 logits 的值均为正，一般做法是对 logits 做一次 softmax，
        这也是为什么 pytorch 默认提供的 nll_loss 实际上不包含 log 操作，
        并将交叉熵分解为 log_softmax 和 nll_loss 两个步骤的原因！

    Examples:
        >>> _logits = torch.randn(5, 5).clamp(min=_EPSILON)  # 负对数似然的输入需要值大于 0
        >>> labels = torch.arange(5)
        >>> _onehot_labels = F.one_hot(labels)
        >>> # 与官方结果比较
        >>> my_ret = negative_log_likelihood_loss(_logits, _onehot_labels)
        >>> official_ret = F.nll_loss(torch.log(_logits + _EPSILON), labels, reduction='none')
        >>> assert torch.allclose(my_ret, official_ret, atol=1e-5)

    Args:
        logits: [B, C], 其中 B 表示 batch_size, C 表示 n_classes
        onehot_labels: same shape as logits
    """
    log_logits = torch.log(logits + _EPSILON)
    return negative_likelihood_loss(log_logits, onehot_labels)


def cross_entropy_loss(probs, onehot_labels):
    """
    交叉熵损失（不带 softmax）

        在不带 softmax 的情况下，交叉熵损失实际上就等价于 `negative_log_likelihood_loss`

        logits -> log -> negative_likelihood_loss

        为什么不带 softmax？——交叉熵损失的输入应该是各类别的概率分布，因此定义上是需要 softmax 的，
            但因为很多时候我们希望模型也输出概率分布，所以通常会对模型的输出做一次 softmax，
            这样在计算 loss 是就不需要再 softmax 了（tensorflow 中就是这样）；

        例如，pytorch 提供的 `nn.CrossEntropyLoss` 就是带有 softmax，那么模型的输出就不需要在
            那么在 eval 时，如果想得到类别的概率分布，还要对结果再做一次 softmax；

    Examples:
        >>> _logits = torch.randn(5, 5).clamp(min=_EPSILON)  # 负对数似然的输入需要值大于 0
        >>> _labels = torch.arange(5)
        >>> _onehot_labels = F.one_hot(_labels)
        >>> # 与官方结果比较
        >>> my_ret = cross_entropy_loss(_logits, _onehot_labels)
        >>> official_ret = F.nll_loss(torch.log(_logits + _EPSILON), _labels, reduction='none')
        >>> assert torch.allclose(my_ret, official_ret, atol=1e-5)

    Args:
        probs:
        onehot_labels:
    """
    return negative_log_likelihood_loss(probs, onehot_labels)


def cross_entropy_loss_with_logits(logits, onehot_labels, dim=-1):
    """ 交叉熵损失（带 softmax），相比 `cross_entropy_loss`，对 logits 多做了一次 softmax

        logits -> softmax -> log -> negative_likelihood

    Examples:
        >>> _logits = torch.randn(5, 5).clamp(min=_EPSILON)  # 负对数似然的输入需要值大于 0
        >>> _labels = torch.arange(5)
        >>> _onehot_labels = F.one_hot(_labels)
        >>> # 与官方结果比较
        >>> my_ret = cross_entropy_loss_with_logits(_logits, _onehot_labels)
        >>> official_ret = F.cross_entropy(_logits, _labels, reduction='none')
        >>> assert torch.allclose(my_ret, official_ret, atol=1e-5)
        >>> # 与 tf 结果比较
        >>> import tensorflow as tf
        >>> os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 屏蔽输出信息，避免影响文档测试的输出
        >>> logits_softmax = tf.nn.softmax(_logits.numpy())
        >>> ce_tf = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        >>> sce_tf = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        >>> assert torch.allclose(my_ret, torch.as_tensor(ce_tf(_onehot_labels, logits_softmax).numpy()), atol=1e-5)
        >>> assert torch.allclose(my_ret, torch.as_tensor(sce_tf(_labels.numpy(), logits_softmax).numpy()), atol=1e-5)

    Args:
        logits:
        onehot_labels:
        dim:

    """
    probs = F.softmax(logits, dim=dim)
    return negative_log_likelihood_loss(probs, onehot_labels)


def sparse_cross_entropy_loss(probs, labels):
    """
    """
    # onehot_labels = F.one_hot(labels, n_classes)
    # return cross_entropy_loss(probs, onehot_labels, eps=eps, dim=dim)
    return F.nll_loss(torch.log(probs), labels, reduction='none')


def sparse_cross_entropy_loss_with_logits(logits, labels):
    """

    Args:
        logits: [B, C], 其中 B 表示 batch_size, C 表示 n_classes
        labels: [B]
    """
    return F.cross_entropy(logits, labels, reduction='none')


def binary_cross_entropy_loss(probs, onehot_labels):
    """
    Examples:
        >>> bce = nn.BCELoss(reduction='none')
        >>> bcel = nn.BCEWithLogitsLoss(reduction='none')
        >>> _logits = torch.rand(3, 2)
        >>> _probs = torch.sigmoid(_logits)  # convert logits to probs
        >>> labels = torch.rand(3, 2)  # shape same as logits
        >>> # 与官方结果比较
        >>> assert torch.allclose(bce(_probs, labels), binary_cross_entropy_loss(_probs, labels), 1e-5)
        >>> assert torch.allclose(bcel(_logits, labels), binary_cross_entropy_loss(_probs, labels), 1e-5)

        # 可见 BCELoss 和 BCEWithLogitsLoss 的区别就是后者自带了 sigmoid 操作

    Args:
        probs:
        onehot_labels:

    Returns:

    """
    return -(onehot_labels * torch.log(probs) + (1 - onehot_labels) * torch.log(1 - probs))


def binary_cross_entropy_loss_with_logits(logits, onehot_labels, dim=-1):
    """"""
    probs = F.softmax(logits, dim=dim)
    return binary_cross_entropy_loss(probs, onehot_labels)


def sparse_binary_cross_entropy_loss(logits, labels):
    """"""
    return F.binary_cross_entropy(logits, labels, reduction='none')


def sparse_binary_cross_entropy_loss_with_logits(logits, labels):
    """"""
    return F.binary_cross_entropy_with_logits(logits, labels, reduction='none')


class CrossEntropyLoss(BaseLoss):
    """@Pytorch Loss
    交叉熵

    区别：官方内置了 softmax 操作，且默认非 one-hot 标签；
    这里参考了 tf 的实现方式

    TODO: 实现 weighted、smooth

    Examples:
        >>> logits = torch.as_tensor([])
        >>> labels = torch.arange(5)
        >>> probs = torch.softmax(logits, dim=-1)
        >>> onehot_labels = F.one_hot(labels)

        >>> my_ce = CrossEntropyLoss(reduction='none')
        >>> ce = nn.CrossEntropyLoss(reduction='none')
        >>> assert torch.allclose(my_ce(probs, onehot_labels), ce(logits, labels), atol=1e-5)

    """

    def __init__(self, from_logits=False, onehot_label=False, multi_label=False, reduction='mean', **base_kwargs):
        super().__init__(reduction=reduction, **base_kwargs)

        # 多标签分类
        if multi_label:
            if onehot_label:
                if from_logits:
                    loss_fn = binary_cross_entropy_loss_with_logits
                else:
                    loss_fn = binary_cross_entropy_loss
            else:
                if from_logits:
                    loss_fn = sparse_binary_cross_entropy_loss_with_logits
                else:
                    loss_fn = sparse_binary_cross_entropy_loss
        # 单标签
        else:
            if onehot_label:
                if from_logits:
                    loss_fn = cross_entropy_loss_with_logits
                else:
                    loss_fn = cross_entropy_loss
            else:
                if from_logits:
                    loss_fn = sparse_cross_entropy_loss_with_logits
                else:
                    loss_fn = sparse_cross_entropy_loss
        self.loss_fn = loss_fn

    def compute_loss(self, inputs, labels):
        return self.loss_fn(inputs, labels)


def _test():
    """"""
    doctest.testmod()


if __name__ == '__main__':
    """"""
    _test()
