#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time: 2021-08-12 4:51 下午

Author: huayang

Subject:

"""
import torch.nn as nn

from my.pytorch.modules.transformer.bert import Bert, get_bert_pretrained
from my.pytorch.modules.wrap import ClassifierWrapper, ClassificationLayer


class BertClassification(nn.Module):
    """"""

    def __init__(self, n_classes, bert: Bert = None, clf_dropout_prob=0.1, problem_type='single'):
        """"""
        super(BertClassification, self).__init__()

        if bert is None:
            bert = get_bert_pretrained()

        self.bert = bert
        self.clf = ClassificationLayer(n_classes=n_classes,
                                       hidden_size=bert.args.hidden_size,
                                       dropout_prob=clf_dropout_prob,
                                       problem_type=problem_type)

    def forward(self, token_ids, token_type_ids=None, masks=None, labels=None):
        """"""
        inputs = {'token_ids': token_ids, 'token_type_ids': token_type_ids, 'masks': masks}
        cls, _, _ = self.bert(**inputs)
        outputs = self.clf(cls, labels=labels)

        if labels is None:  # eval
            probs = outputs
            return probs
        else:
            probs, loss = outputs
            return probs, loss
