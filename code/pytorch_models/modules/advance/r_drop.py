#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time: 2021-10-11 4:02 下午

Author: huayang

Subject:

"""
import doctest

import torch.nn as nn

from my.pytorch.loss.r_drop import RDropLoss


class RDrop(nn.Module):
    """"""

    def __init__(self, encoder, **loss_kwargs):
        """"""
        super().__init__()

        self.encoder = encoder
        self.loss_fn = RDropLoss(**loss_kwargs)

    def forward(self, inputs, labels=None):
        """"""
        logits1 = self.encoder(**inputs)

        if labels is not None:
            logits2 = self.encoder(**inputs)
            loss = self.loss_fn(logits1, logits2, labels)

            return logits1, logits2, loss

        return logits1


def _test():
    """"""
    doctest.testmod()


if __name__ == '__main__':
    """"""
    _test()
