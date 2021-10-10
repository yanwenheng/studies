#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time:
   2021-07-20 12:39 下午
   
Author:
   huayang
   
Subject:
   
"""
from torch import Tensor

from my.pytorch.modules.transformer.bert import get_bert_pretrained
from my.pytorch.modules.sequence_labeling.crf import CRFEncoder


class BertCRF(CRFEncoder):
    """@Pytorch Models
    BertCRF
    """

    def __init__(self, n_classes, bert=None, **kwargs):
        if bert is None:
            bert = get_bert_pretrained()
        super(BertCRF, self).__init__(n_classes, encoder=bert, **kwargs)

    def encode_wrapper(self, encoder, inputs) -> Tensor:
        return super().encode_wrapper(encoder, inputs)[1]


def _test():
    """"""

    def _test_BertCRF():  # noqa
        """"""
        BertCRF(3)

    _test_BertCRF()


if __name__ == '__main__':
    """"""
    _test()
