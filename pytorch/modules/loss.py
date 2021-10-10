#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time:
    2021-07-02 10:19 上午
    
Author:
    huayang
    
Subject:
    
"""
import doctest

import torch
import torch.nn as nn
import torch.nn.functional as F

from my.pytorch.backend.distance_fn import euclidean_distance, euclidean_distance_nosqrt

from my.pytorch.backend.loss_fn import cross_entropy_loss
from my.pytorch.backend.loss_fn import contrastive_loss
from my.pytorch.backend.loss_fn import triplet_loss

LABEL_FORMATS = {'category', 'one_hot'}


__all__ = [
    'BaseLoss',
    'ContrastiveLoss',
    'TripletLoss',
    'CrossEntropyLoss'
]

class BaseLoss(nn.Module):
    """@Pytorch Loss
    Loss 基类
    """

    def __init__(self, reduction='mean', temperature=1.0):
        """
        注意继承类的 `super().__init__(**kwargs)` 需要放在 `__init__()` 的最后执行

        Examples:
            class CrossEntropyLoss(BaseLoss):
                def __init__(self, eps=1e-8, **kwargs):
                    self.loss_fn = lambda inputs, labels: cross_entropy_loss(inputs, labels, eps=eps)
                    super(CrossEntropyLoss, self).__init__(**kwargs)

        Args:
            reduction: 约减类型，以下三者之一：{'mean', 'sum', 'none'}
            temperature: 对 loss 进行缩放，值在 (0, +] 之间，即 `loss = loss / temperature`
        """
        super(BaseLoss, self).__init__()

        if reduction == 'none':
            reduce_fn = lambda x: x
        elif reduction == 'mean':
            reduce_fn = torch.mean
        elif reduction == 'sum':
            reduce_fn = torch.sum
        else:
            raise ValueError(f"reduction={reduction} not in ('mean', 'sum', 'none').")
        self.reduce_fn = reduce_fn

        self.temperature = temperature
        assert self.temperature > 0, f'`temperature` should greater than 0, but {self.temperature}'

        # assert self.loss_fn
        # assert isinstance(self.loss_fn, Callable), f'`loss_fn` must be callable, but {type(self.loss_fn)}'

    def compute_loss(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        """"""
        loss = self.compute_loss(*args, **kwargs)
        loss = self.reduce_fn(loss)  # reduction
        loss /= self.temperature  # scale
        return loss


class ContrastiveLoss(BaseLoss):
    """@Pytorch Loss
    对比损失（可自定义距离函数，默认为欧几里得距离）
    """

    def __init__(self, distance_fn=euclidean_distance, margin=1.0, **kwargs):
        """"""
        self.margin = margin
        self.distance_fn = distance_fn

        super(ContrastiveLoss, self).__init__(**kwargs)

    def compute_loss(self, x1, x2, labels):
        return contrastive_loss(x1, x2, labels,
                                distance_fn=self.distance_fn, margin=self.margin)


class TripletLoss(BaseLoss):
    """@Pytorch Loss
    Triplet 损失，常用于无监督学习、few-shot 学习

    Examples:
        >>> anchor = torch.randn(100, 128)
        >>> positive = torch.randn(100, 128)
        >>> negative = torch.randn(100, 128)

        # my_tl 默认 euclidean_distance_nosqrt
        >>> tl = TripletLoss(margin=2., reduction='none')
        >>> tld = nn.TripletMarginWithDistanceLoss(distance_function=euclidean_distance_nosqrt, margin=2., reduction='none')
        >>> assert torch.allclose(tl(anchor, positive, negative), tld(anchor, positive, negative), atol=1e-5)

        # 自定义距离函数
        >>> from my.pytorch.backend.distance_fn import cosine_distance
        >>> my_tl = TripletLoss(distance_fn=cosine_distance, margin=0.5, reduction='none')
        >>> tl = nn.TripletMarginWithDistanceLoss(distance_function=cosine_distance, margin=0.5, reduction='none')
        >>> assert torch.allclose(my_tl(anchor, positive, negative), tl(anchor, positive, negative), atol=1e-5)

    """

    def __init__(self, distance_fn=euclidean_distance_nosqrt, margin=1.0, **kwargs):
        """"""
        self.margin = margin
        self.distance_fn = distance_fn

        super(TripletLoss, self).__init__(**kwargs)

    def compute_loss(self, anchor, positive, negative):
        return triplet_loss(anchor, positive, negative,
                            distance_fn=self.distance_fn, margin=self.margin)


class CrossEntropyLoss(BaseLoss):
    """@Pytorch Loss
    交叉熵

    区别：官方内置了 softmax 操作，且默认非 one-hot 标签；
    这里参考了 tf 的实现方式

    Examples:
        >>> logits = torch.randn(5, 5)
        >>> labels = torch.arange(5)
        
        >>> probs = torch.softmax(logits, dim=-1)
        >>> onehot_labels = F.one_hot(labels)

        >>> my_ce = CrossEntropyLoss(reduction='none')
        >>> ce = nn.CrossEntropyLoss(reduction='none')
        >>> assert torch.allclose(my_ce(probs, onehot_labels), ce(logits, labels), atol=1e-5)

    """

    def __init__(self, **kwargs):
        """"""
        super(CrossEntropyLoss, self).__init__(**kwargs)

    def compute_loss(self, inputs, labels):
        return cross_entropy_loss(inputs, labels)


def _test():
    """"""
    doctest.testmod()


if __name__ == '__main__':
    """"""

    _test()
