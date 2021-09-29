#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time:
    2021-06-30 5:32 下午
    
Author:
    huayang
    
Subject:
    
"""

import torch.nn as nn
import torch.nn.functional as F

# use full path, avoid circular reference
from my.pytorch.modules.transformer.multi_head_attention import MultiHeadAttention

__all__ = [
    'TransformerBlock',
]


class TransformerBlock(nn.Module):
    """ Attention is all you need """

    def __init__(self,
                 hidden_size=768,
                 intermediate_size=3072,
                 num_attention_heads=12,
                 activation_fn=F.gelu,
                 dropout_prob=0.1,
                 attention_dropout_prob=0.1,
                 layer_norm_eps=1e-12,
                 pre_ln=False):
        """"""
        super().__init__()

        self.attention = MultiHeadAttention(hidden_size, num_attention_heads, dropout_prob=attention_dropout_prob)
        self.attention_ln = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        self.ffn = FeedForward(hidden_size, intermediate_size, activation_fn)
        self.ffn_ln = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        self.dropout = nn.Dropout(dropout_prob)

        """
        Post-LN: Attn -> Drop -> Add -> LN -> FFN -> Drop -> Add -> LN
        Pre-LN:  LN -> Attn -> Add -> Drop -> LN -> FFN -> Add -> Drop
        """
        self.post_ln = pre_ln
        if pre_ln:
            self.forward = self.forward_pre
        else:  # default
            self.forward = self.forward_post

    def forward_post(self, inputs, masks):
        """"""
        x = self.attention(inputs, masks)
        x = self.dropout(x) + inputs
        x = self.attention_ln(x)

        inputs = x[:]
        x = self.ffn(x)
        x = self.dropout(x) + inputs
        x = self.ffn_ln(x)

        return x

    def forward_pre(self, inputs, masks):
        """"""
        x = self.attention_ln(inputs)
        x = self.attention(x, masks)
        x = self.dropout(x) + inputs

        inputs = x[:]
        x = self.ffn_ln(inputs)
        x = self.ffn(x)
        x = self.dropout(x) + inputs

        return x


class FeedForward(nn.Module):
    """ Position Wise Feed Forward """

    def __init__(self, hidden_size, intermediate_size, activation_fn):
        super().__init__()

        self.W1 = nn.Linear(hidden_size, intermediate_size)
        self.W2 = nn.Linear(intermediate_size, hidden_size)
        self.act = activation_fn

    def forward(self, inputs):
        """"""
        x = self.W2(self.act(self.W1(inputs)))
        return x
