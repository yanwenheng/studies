#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time:
    2021-06-27 8:32 下午
    
Author:
    huayang
    
Subject:
    
"""
import math

import torch
import torch.nn as nn

__all__ = [
    'MultiHeadAttention',
]


class MultiHeadAttention(nn.Module):
    """"""

    def __init__(self,
                 hidden_size=768,
                 num_attention_heads=12,
                 hidden_size_per_head=None,  # default 64=768/12
                 dropout_prob=0.1):
        """"""
        super().__init__()
        # default 768=64*12
        hidden_size = hidden_size or hidden_size_per_head * num_attention_heads
        assert hidden_size % num_attention_heads == 0, 'hidden_size % num_attention_heads != 0'

        self.num_attention_heads = num_attention_heads
        self.hidden_size_per_head = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.hidden_size_per_head  # equal to `hidden_size`

        self.q_dense = nn.Linear(hidden_size, self.all_head_size)
        self.k_dense = nn.Linear(hidden_size, self.all_head_size)
        self.v_dense = nn.Linear(hidden_size, self.all_head_size)
        self.o_dense = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(dropout_prob)
        self.softmax = nn.Softmax(-1)

    def forward(self, inputs, masks=None, output_attentions=False):
        """
        Args:
            inputs: [batch_size, seq_len, hidden_size]
            masks: [batch_size, seq_len]
            output_attentions:

        Returns:
            output: [batch_size, seq_len, hidden_size]
            or
            (output, attentions)
        """
        if isinstance(inputs, (list, tuple)):
            q, k, v = inputs[:3]
        else:
            q = k = v = inputs  # [batch_size, seq_len, hidden_size]

        # mask assert
        if masks.ndim == 2:  # [batch_size, seq_len]
            masks = masks[:, None, None, :]  # -> [B, 1, 1, seq_len]
        elif masks.ndim == 3:  # [batch_size, seq_len_from, seq_len_to]
            masks = masks[:, None, :, :]  # -> [batch_size, 1, seq_len_from, seq_len_to]
        else:
            raise ValueError(f'Error mask ndim={masks.ndim}, it should be 2 or 3.')

        # dims
        B = q.shape[0]  # batch_size
        Q = q.shape[1]  # seq_len_from (query)
        V = v.shape[1]  # seq_len_to (key, value)
        N = self.hidden_size_per_head
        H = self.num_attention_heads

        # multi-head linear
        q = self.q_dense(q).reshape([B, Q, H, N]).transpose(1, 2)  # [B, H, Q, N]
        k = self.k_dense(k).reshape([B, V, H, N]).transpose(1, 2)  # [B, H, V, N]
        v = self.v_dense(v).reshape([B, V, H, N]).transpose(1, 2)  # [B, H, V, N]

        # multi-head scaled dot-product attention
        a = torch.matmul(q, k.transpose(-1, -2))  # [B, H, Q, N] x [B, H, N, V] -> [B, H, Q, V]
        a = a / math.sqrt(self.hidden_size_per_head)  # scale
        a = a.masked_fill(masks == 0, -1e12) if masks is not None else a  # mask
        a = self.softmax(a)  # [B, H, Q, V]
        a = self.dropout(a)  # [B, H, Q, V]

        # outputs
        o = torch.matmul(a, v)  # [B, H, Q, V] x [B, H, V, N] -> [B, H, Q, N]
        o = o.transpose(1, 2).reshape([B, Q, H * N])  # [B, H, Q, N] -> [B, Q, H, N] -> [B, Q, H*N]
        o = self.o_dense(o)  # linear

        outputs = (o, a) if output_attentions else o
        return outputs


def _test():
    """"""

    def _test_self_attention():
        """自注意力"""
        # batch_size=2, seq_len=5
        inputs_id = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]])
        mask = (inputs_id != 0).to(torch.float)
        mask = mask[:, None, None, :]  # for broadcast

        # vocab_size=6, embed_size=768
        embed = nn.Embedding(6, 768)
        inputs = embed(inputs_id)

        # attention
        self_attention = MultiHeadAttention()
        outputs = self_attention(inputs, mask)
        print(outputs.shape)  # [2, 5, 768]

    _test_self_attention()


if __name__ == '__main__':
    """"""
    _test()
