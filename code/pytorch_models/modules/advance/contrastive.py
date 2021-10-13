#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time:
    2021-09-03 6:32 下午

Author:
    huayang

Subject:
    对比学习：
        - 孪生网络
        - Triplet 网络
        - SimCSE
"""

# 孪生网络
from pytorch_models.modules.advance.siamese import SiameseNet
# Triplet
from pytorch_models.modules.advance.triplet import TripletNet
# SimCSE
from pytorch_models.modules.advance.sim_cse import SimCSE


def _test():
    """"""

    def _test_base():
        """"""
        from pytorch_models.modules.transformer.bert import get_bert_pretrained
        bert = get_bert_pretrained()

        sn = SiameseNet(encoder=bert)
        tn = TripletNet(encoder=bert)
        sc = SimCSE(encoder=bert)


if __name__ == '__main__':
    """"""
    _test()
