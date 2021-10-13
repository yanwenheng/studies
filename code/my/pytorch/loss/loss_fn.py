#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time:
    2021-08-05 4:56 下午

Author:
    huayang

Subject:

"""
import doctest

import torch
import torch.nn.functional as F  # noqa

from my.pytorch.loss.BaseLoss import _EPSILON


def mean_squared_error_loss(inputs, targets):
    """ 平方差损失

    Examples:
        >>> i = torch.randn(3, 5)
        >>> t = torch.randn(3, 5)

        # 与官方结果比较
        >>> my_ret = mean_squared_error_loss(i, t)
        >>> official_ret = F.mse_loss(i, t, reduction='none')
        >>> assert torch.allclose(my_ret, official_ret, atol=1e-5)

    Args:
        inputs: [B, N]
        targets: same shape as inputs

    Returns:
        [B, N]
    """
    return (inputs - targets).pow(2.0)


def cosine_similarity_loss(x1, x2, labels):
    """ cosine 相似度损失

    Examples:
        # >>> logits = torch.randn(5, 5).clamp(min=_EPSILON)  # 负对数似然的输入需要值大于 0
        # >>> labels = torch.arange(5)
        # >>> onehot_labels = F.one_hot(labels)
        #
        # # 与官方结果比较
        # >>> my_ret = negative_log_likelihood_loss(logits, onehot_labels)
        # >>> official_ret = F.nll_loss(torch.log(logits + _EPSILON), labels, reduction='none')
        # >>> assert torch.allclose(my_ret, official_ret, atol=1e-5)

    Args:
        x1: [B, N]
        x2: same shape as x1
        labels: [B] or scalar

    Returns:
        [B] vector or scalar
    """
    cosine_scores = F.cosine_similarity(x1, x2, dim=-1)  # [B]
    return mean_squared_error_loss(cosine_scores, labels)  # [B]


def kl_div_loss(p, q, masks=None):
    """"""
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')

    # pad_mask is for seq-level tasks
    if masks is not None:
        p_loss.masked_fill_(masks, 0.)
        q_loss.masked_fill_(masks, 0.)

    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.sum()
    q_loss = q_loss.sum()

    loss = (p_loss + q_loss) / 2
    return loss


def _test():
    """"""
    doctest.testmod()


if __name__ == '__main__':
    """"""
    _test()
