#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time:
    2021-08-05 4:56 下午

Author:
    huayang

Subject:

"""
import os
import doctest

import torch
import torch.nn as nn
import torch.nn.functional as F

from my.pytorch.backend.distance_fn import euclidean_distance

_EPSILON = 1e-8


def contrastive_loss(x1, x2, labels, distance_fn=euclidean_distance, margin=2.0):
    """ 对比损失 (0 <= label <= 1)
        - 当 y=1（即样本相似）时，如果距离较大，则加大损失；
        - 当 y=0（即样本不相似）时，如果距离反而小，也会增大损失；

    Args:
        x1:
        x2:
        labels:
        distance_fn: 默认为欧几里得距离
        margin: 需要根据使用距离函数调整

    Returns:

    """
    labels = labels.float()
    distances = distance_fn(x1, x2)
    return 0.5 * (labels * distances.pow(2) + (1 - labels) * F.relu(margin - distances).pow(2))


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


def negative_likelihood_loss(logits, onehot_labels):
    """ 负似然损失，等价于 `F.nll_loss()`

        logits -> negative_likelihood_loss

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
    """ 负对数似然损失，相比 `negative_likelihood_loss`，在计算损失之前，先对 `logits` 计算一次 `log`

        logits -> log -> negative_likelihood_loss

        注意：因为要计算一次 log，所以需确保 logits 的值均为正，所以一般会提前对 logits 做一次 softmax，
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
    logits_log = torch.log(logits + _EPSILON)
    return negative_likelihood_loss(logits_log, onehot_labels)


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


def cross_entropy_loss(probs, onehot_labels):
    """ 交叉熵损失（不带 softmax）
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


def cross_entropy_softmax_loss(logits, onehot_labels, dim=-1):
    """ 交叉熵损失（带 softmax），相比 `cross_entropy_loss`，对 logits 多做了一次 softmax

        logits -> softmax -> log -> negative_likelihood

    Examples:
        >>> _logits = torch.randn(5, 5).clamp(min=_EPSILON)  # 负对数似然的输入需要值大于 0
        >>> _labels = torch.arange(5)
        >>> _onehot_labels = F.one_hot(_labels)
        >>> # 与官方结果比较
        >>> my_ret = cross_entropy_softmax_loss(_logits, _onehot_labels)
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
    probs = torch.softmax(logits, dim=dim)
    return negative_log_likelihood_loss(probs, onehot_labels)


def cross_entropy_sparse_loss(probs, labels):
    """ 等价于没有 softmax 的 `F.cross_entropy`
    """
    # onehot_labels = F.one_hot(labels, n_classes)
    # return cross_entropy_loss(probs, onehot_labels, eps=eps, dim=dim)
    return F.nll_loss(torch.log(probs), labels, reduction='none')


def cross_entropy_sparse_softmax_loss(logits, labels):
    """"""
    return F.cross_entropy(logits, labels, reduction='none')


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


def triplet_loss(anchor, positive, negative, distance_fn=F.pairwise_distance, margin=2.0):
    """  triplet 损失

    Examples:
        >>> a = torch.randn(100, 128)
        >>> p = torch.randn(100, 128)
        >>> n = torch.randn(100, 128)
        >>> # 官方提供的 triplet_loss
        >>> tl = nn.TripletMarginLoss(margin=2.0, p=2, reduction='none')
        >>> assert torch.allclose(triplet_loss(a, p, n), tl(a, p, n), atol=1e-5)
        >>> # 官方提供的 triplet_loss: 自定义距离函数
        >>> from my.pytorch.backend.distance_fn import cosine_distance
        >>> tld = nn.TripletMarginWithDistanceLoss(distance_function=cosine_distance, margin=2.0, reduction='none')
        >>> assert torch.allclose(triplet_loss(a, p, n, distance_fn=cosine_distance), tld(a, p, n), atol=1e-5)

    Args:
        anchor:
        positive:
        negative:
        distance_fn:
        margin:

    Returns:
        [B]

    Examples:
        anchor = torch.randn(100, 128)
        positive = torch.randn(100, 128)
        negative = torch.randn(100, 128)

        # 自定义距离
        from my.pytorch.backend.distance_fn import cosine_distance

        # 官方
        tld = nn.TripletMarginWithDistanceLoss(distance_function=cosine_distance, margin=2.0, reduction='none')
        o = tld(anchor, positive, negative)

        # my
        triplet_loss(anchor, positive, negative, distance_fn=cosine_distance)

    """
    distance_pos = distance_fn(anchor, positive)
    distance_neg = distance_fn(anchor, negative)
    return torch.relu(distance_pos - distance_neg + margin)


def _test():
    """"""
    doctest.testmod()


if __name__ == '__main__':
    """"""
    _test()
