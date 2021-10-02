#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time:
    2021-06-30 8:14 下午
    
Author:
    huayang
    
Subject:
    
"""
import os
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from my.nlp.bert_tokenizer import Tokenizer
from my.python.utils import set_env, get_env
from my.python.config_loader import load_config_file
from my.python.custom import ConfigDict
from my.pytorch.utils import get_state_dict, load_weights_partly
from my.pytorch.backend.activation_fn import ACT_STR2FN
from my.pytorch.modules.transformer.transformer_block import TransformerBlock

__all__ = [
    'Bert',
    'BertConfig',
    'get_bert_pretrained',
    'get_CKPT_DIR',
    'set_CKPT_DIR'
]


def set_CKPT_DIR(ckpt_dir):  # noqa
    """"""
    set_env('CKPT', ckpt_dir)


def get_CKPT_DIR():  # noqa
    return get_env('CKPT', os.path.join(os.environ['HOME'], 'workspace/ckpt'))


class BertConfig(ConfigDict):

    def __init__(self, **config_items):
        """ Default Base Config """
        self.hidden_size = 768  # hidden_size
        self.vocab_size = 21128  # chinese vocab
        self.intermediate_size = 3072  # 768 * 4
        self.num_hidden_layers = 12  # num_transformers
        self.num_attention_heads = 12  # num_attention_heads
        self.max_position_embeddings = 512  # max_seq_len
        self.type_vocab_size = 2  # num_token_type
        self.hidden_act = 'gelu'  # activation_fn
        self.hidden_dropout_prob = 0.1  # dropout_prob
        self.attention_probs_dropout_prob = 0.1  # attention_dropout_prob
        self.initializer_range = 0.02  # normal_std

        # no use
        self.directionality = "bidi"
        self.pooler_fc_size = 768
        self.pooler_num_attention_heads = 12
        self.pooler_num_fc_layers = 3
        self.pooler_size_per_head = 128
        self.pooler_type = "first_token_transform"

        # custom
        self.pre_ln = False

        super(BertConfig, self).__init__(allow_new_item=True, **config_items)


class BertEmbedding(nn.Module):
    """ Bert Embedding """

    def __init__(self,
                 vocab_size=21128,
                 embedding_size=768,
                 max_seq_len=512,
                 num_token_types=2,
                 pad_token_id=0,
                 dropout_prob=0.1,
                 layer_norm_eps=1e-12):
        """"""
        super().__init__()
        self.max_seq_len = max_seq_len

        self.word_embeddings = nn.Embedding(vocab_size, embedding_size, padding_idx=pad_token_id)
        self.position_embeddings = nn.Embedding(max_seq_len, embedding_size)
        self.token_type_embeddings = nn.Embedding(num_token_types, embedding_size)

        self.layer_norm = nn.LayerNorm(embedding_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout_prob)

        # not model parameters
        self.register_buffer("position_ids", torch.arange(max_seq_len).expand((1, -1)))

    def forward(self, token_ids, segment_ids):
        """"""
        seq_length = token_ids.shape[1]
        assert seq_length <= self.max_seq_len

        word_embeddings = self.word_embeddings(token_ids)
        token_type_embeddings = self.token_type_embeddings(segment_ids)

        position_ids = self.position_ids[:, :seq_length]
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = word_embeddings + token_type_embeddings + position_embeddings  # add
        embeddings = self.layer_norm(embeddings)  # layer_norm
        embeddings = self.dropout(embeddings)  # dropout
        return embeddings


class BertPooler(nn.Module):
    """ Bert Pooler """

    def __init__(self, hidden_size=768, activation_fn=torch.tanh):
        """"""
        super().__init__()

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.act = activation_fn

    def forward(self, inputs):
        """"""
        return self.act(self.dense(inputs))


class Bert(nn.Module):
    """ Bert """

    def __init__(self,
                 args: BertConfig = None,
                 init_weights: bool = False):
        """"""
        super().__init__()

        # init args
        if args is None:
            args = BertConfig()
        self.args = args
        self._init_args()

        # build model
        self.embeddings = BertEmbedding(**self._embedding_args)
        self.transformers = nn.ModuleList([TransformerBlock(**self._transformer_args)
                                           for _ in range(self._num_transformers)])
        self.pooler = BertPooler(args.hidden_size)

        # init weights
        if init_weights:
            self.apply(self._init_fn)  # recursive init all modules

    def forward(self, token_ids: Tensor, token_type_ids=None, masks=None):
        """"""
        # token_type_ids
        if token_type_ids is None:  # 默认输入是单句
            token_type_ids = torch.zeros(token_ids.shape, dtype=token_ids.dtype)

        # compute mask
        if masks is None:  # 默认根据 token_ids 推断
            masks = (token_ids > 0).to(torch.uint8)  # [B, L]

        # embedding
        x = self.embeddings(token_ids, token_type_ids)

        # transformers
        all_hidden_states = [x]
        for transformer in self.transformers:
            x = transformer(x, masks)  # [B, L, N]
            all_hidden_states.append(x)

        # pooler
        hidden_states = all_hidden_states[-1]  # [B, L, N]
        cls_before_pooler = hidden_states[:, 0]  # [B, N]
        cls = self.pooler(cls_before_pooler)  # [B, N]

        return cls, hidden_states, all_hidden_states

    def get_input_embeddings(self):
        """"""
        return self.embeddings.word_embeddings

    def _init_args(self):
        """"""
        args = self.args
        self._normal_std = args.initializer_range
        self._num_transformers = args.num_hidden_layers
        self._embedding_args = {
            'vocab_size': args.vocab_size,
            'embedding_size': args.hidden_size,
            'max_seq_len': args.max_position_embeddings,
            'num_token_types': args.type_vocab_size,
            'dropout_prob': args.hidden_dropout_prob,
        }
        self._transformer_args = {
            'hidden_size': args.hidden_size,
            'intermediate_size': args.intermediate_size,
            'num_attention_heads': args.num_attention_heads,
            'activation_fn': ACT_STR2FN[args.hidden_act] if isinstance(args.hidden_act, str) else args.hidden_act,
            'dropout_prob': args.hidden_dropout_prob,
            'attention_dropout_prob': args.attention_probs_dropout_prob,
            'pre_ln': args.pre_ln,
        }

    def _init_fn(self, module: nn.Module):
        """"""
        _init_weights(module, normal_std=self._normal_std)


# TODO
class _BertForPreTrain(nn.Module):
    """"""

    def __init__(self, **kwargs):
        """"""
        super().__init__()

        self.bert = Bert(**kwargs)

        # hidden_size = kwargs.get('hidden_size', 768)
        # vocab_size = kwargs.get('vocab_size', 21128)
        # mlm = _BertMaskedLanguageModel(hidden_size, vocab_size)
        # nsp = _BertNextSentencePrediction(hidden_size)
        # self.cls = ScopeWrapper(mlm=mlm, nsp=nsp)

    def forward(self, inputs):
        """"""
        cls_embedding, all_token_state = self.bert(*inputs)
        mlm_score = self.mlm(all_token_state)
        nsp_score = self.nsp(cls_embedding)
        return mlm_score, nsp_score


class _BertMaskedLanguageModel(nn.Module):
    """ Bert MLM task """

    def __init__(self,
                 hidden_size=768,
                 vocab_size=21128,
                 activation_fn=F.gelu,
                 layer_norm_eps=1e-12):
        super().__init__()

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.act = activation_fn
        # TODO: 如何将 decoder 的权重替换为 word_embeddings
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)  # same as word_embedding.T
        self.decoder.bias = nn.Parameter(torch.zeros(vocab_size))
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.softmax = nn.LogSoftmax(-1)

    def forward(self, inputs):
        """"""
        x = self.act(self.dense(inputs))
        x = self.layer_norm(x)
        x = self.decoder(x)
        return x


class _BertNextSentencePrediction(nn.Module):
    """ Bert NSP task """

    def __init__(self, hidden_size=768):
        super().__init__()

        self.dense = nn.Linear(hidden_size, 2)
        self.softmax = nn.LogSoftmax(-1)

    def forward(self, inputs):
        x = self.dense(inputs)
        prob = self.softmax(x)
        return prob


def _init_weights(module: nn.Module, normal_std=0.02):
    """"""
    if isinstance(module, nn.Linear):
        # truncated_normal
        nn.init.trunc_normal_(module.weight.data, std=normal_std)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        # truncated_normal
        nn.init.trunc_normal_(module.weight.data, std=normal_std)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.LayerNorm):
        module.weight.data.fill_(1.0)
        module.bias.data.zero_()
    else:
        pass  # default


# 权重对应关系：my_bert -> tf_google_bert
WEIGHTS_NAME_MAP_TF = {
    # embeddings
    'embeddings.word_embeddings.weight': 'bert/embeddings/word_embeddings',
    'embeddings.position_embeddings.weight': 'bert/embeddings/position_embeddings',
    'embeddings.token_type_embeddings.weight': 'bert/embeddings/token_type_embeddings',
    'embeddings.layer_norm.weight': 'bert/embeddings/LayerNorm/gamma',
    'embeddings.layer_norm.bias': 'bert/embeddings/LayerNorm/beta',

    # transformer
    'transformers.{idx}.attention.q_dense.weight': 'bert/encoder/layer_{idx}/attention/self/query/kernel',
    'transformers.{idx}.attention.q_dense.bias': 'bert/encoder/layer_{idx}/attention/self/query/bias',
    'transformers.{idx}.attention.k_dense.weight': 'bert/encoder/layer_{idx}/attention/self/key/kernel',
    'transformers.{idx}.attention.k_dense.bias': 'bert/encoder/layer_{idx}/attention/self/key/bias',
    'transformers.{idx}.attention.v_dense.weight': 'bert/encoder/layer_{idx}/attention/self/value/kernel',
    'transformers.{idx}.attention.v_dense.bias': 'bert/encoder/layer_{idx}/attention/self/value/bias',
    'transformers.{idx}.attention.o_dense.weight': 'bert/encoder/layer_{idx}/attention/output/dense/kernel',
    'transformers.{idx}.attention.o_dense.bias': 'bert/encoder/layer_{idx}/attention/output/dense/bias',
    'transformers.{idx}.attention_ln.weight': 'bert/encoder/layer_{idx}/attention/output/LayerNorm/gamma',
    'transformers.{idx}.attention_ln.bias': 'bert/encoder/layer_{idx}/attention/output/LayerNorm/beta',
    'transformers.{idx}.ffn.W1.weight': 'bert/encoder/layer_{idx}/intermediate/dense/kernel',
    'transformers.{idx}.ffn.W1.bias': 'bert/encoder/layer_{idx}/intermediate/dense/bias',
    'transformers.{idx}.ffn.W2.weight': 'bert/encoder/layer_{idx}/output/dense/kernel',
    'transformers.{idx}.ffn.W2.bias': 'bert/encoder/layer_{idx}/output/dense/bias',
    'transformers.{idx}.ffn_ln.weight': 'bert/encoder/layer_{idx}/output/LayerNorm/gamma',
    'transformers.{idx}.ffn_ln.bias': 'bert/encoder/layer_{idx}/output/LayerNorm/beta',

    # pooler
    'pooler.dense.weight': 'bert/pooler/dense/kernel',
    'pooler.dense.bias': 'bert/pooler/dense/bias',

    # TODO: task 部分
    'cls.mlm.dense.weight': 'cls/predictions/transform/dense/kernel',
    'cls.mlm.dense.bias': 'cls/predictions/transform/dense/bias',
    'cls.mlm.layer_norm.weight': 'cls/predictions/transform/LayerNorm/gamma',
    'cls.mlm.layer_norm.bias': 'cls/predictions/transform/LayerNorm/beta',
    'cls.mlm.decoder.bias': 'cls/predictions/output_bias',
    'cls.nsp.dense.weight': 'cls/seq_relationship/output_weights',
    'cls.nsp.dense.bias': 'cls/seq_relationship/output_bias',
}


def _build_name_mapping_tf(num_transformers=12):
    """"""
    mapping_dict = dict()
    for name_f, name_t in WEIGHTS_NAME_MAP_TF.items():
        if re.search(r'idx', name_f):
            for idx in range(num_transformers):
                name_f_idx = name_f.format(idx=idx)
                name_t_idx = name_t.format(idx=idx)
                mapping_dict[name_f_idx] = name_t_idx
        else:
            mapping_dict[name_f] = name_t

    return mapping_dict


def build_name_mapping(num_transformers, from_tf=True, with_prefix=False):
    """"""
    name_mapping = _build_name_mapping_tf(num_transformers)
    if from_tf:
        return name_mapping

    for name_f, name_t in name_mapping.items():
        if not with_prefix:
            name_t = name_t[5:]

        name_t = name_t.replace('/', '.')
        name_t = name_t.replace('layer_', 'layer.')
        name_t = name_t.replace('kernel', 'weight')
        name_t = name_t.replace('gamma', 'weight')
        name_t = name_t.replace('beta', 'bias')

        if name_t.endswith('_embeddings'):
            name_t += '.weight'

        name_mapping[name_f] = name_t

    return name_mapping


def get_bert_pretrained(model_dir: str = None,
                        name_mapping: dict = None,
                        config_file_name: str = None,
                        weight_file_name: str = None,
                        vocab_file_name: str = None,
                        return_tokenizer=False,
                        from_tf=True):
    """"""
    if model_dir is None:
        model_dir = os.path.join(get_CKPT_DIR(), 'chinese_L-12_H-768_A-12')

    if from_tf:
        config_file_name = config_file_name or 'bert_config.json'
        weight_file_name = weight_file_name or 'bert_model.ckpt'
    else:
        config_file_name = config_file_name or 'config.json'
        weight_file_name = weight_file_name or 'pytorch_model.bin'

    config_path = os.path.join(model_dir, config_file_name)
    weight_path = os.path.join(model_dir, weight_file_name)

    # build model
    args = load_config_file(config_path, cls=BertConfig)
    bert = Bert(args)

    # load weights
    weights_dict = get_state_dict(weight_path, from_tf=from_tf)
    name_mapping = name_mapping or build_name_mapping(args.num_hidden_layers, from_tf=from_tf, with_prefix=True)
    load_weights_partly(bert, weights_dict, name_mapping)

    # build tokenizer
    if return_tokenizer:
        vocab_file_name = vocab_file_name or 'vocab.txt'
        vocab_path = os.path.join(model_dir, vocab_file_name)
        tokenizer = Tokenizer(vocab_path)
        return bert, tokenizer

    return bert


def _test():
    """"""

    def _test_basic():
        """"""
        from transformers import BertModel
        from my.nlp.bert_tokenizer import tokenizer
        s = '我爱机器学习'
        tid, sid, mask = tokenizer.encode(s, max_len=10)
        # print(tid, sid)
        tids = torch.tensor([tid])
        sids = torch.tensor([sid])
        masks = torch.tensor([mask])

        # transformers Bert
        model = BertModel.from_pretrained('bert-base-chinese')
        model.config.output_hidden_states = True
        o_pt = model(tids, masks, sids)

        # My bert 1 (default)
        bert = get_bert_pretrained(return_tokenizer=False)
        o_my = bert(tids, sids)
        # cls embedding
        assert torch.allclose(o_pt.pooler_output, o_my[0], atol=1e-5)
        # last_hidden_state
        assert torch.allclose(o_pt.last_hidden_state, o_my[1], atol=1e-5)
        # all_hidden_state
        assert len(o_pt.hidden_states) == len(o_my[-1]) == 13
        assert torch.allclose(torch.cat(o_pt.hidden_states), torch.cat(o_my[-1]), atol=1e-5)

        # My bert 2 (build from pt weights)
        bert = Bert()
        weights_path = r'/Users/huayang/.cache/huggingface/transformers/58592490276d9ed1e8e33f3c12caf23000c22973cb2b3218c641bd74547a1889.fabda197bfe5d6a318c2833172d6757ccc7e49f692cb949a6fabf560cee81508'
        sd = get_state_dict(weights_path, from_tf=False)
        # for k, v in sd.items():
        #     print(k, v.shape)
        # # load_weights_from_pt(bert, weights_path)
        name_mapping = build_name_mapping(num_transformers=12, from_tf=False, with_prefix=True)
        load_weights_partly(bert, sd, name_mapping)
        # load_weights_from_others(bert, weights_path, name_mapping)
        o_my = bert(tids)
        # cls embedding
        assert torch.allclose(o_pt.pooler_output, o_my[0], atol=1e-5)
        # last_hidden_state
        assert torch.allclose(o_pt.last_hidden_state, o_my[1], atol=1e-5)
        # all_hidden_state
        assert torch.allclose(torch.cat(o_pt.hidden_states), torch.cat(o_my[-1]), atol=1e-5)

    _test_basic()


if __name__ == '__main__':
    """"""
    _test()
