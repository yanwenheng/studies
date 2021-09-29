#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time:
   2021-07-20 5:05 下午
   
Author:
   huayang
   
Subject:
   序列标注型数据处理
"""
import os

from sortedcollections import OrderedSet

from typing import *

from my.python.utils import get_logger

logger = get_logger(__name__)

__all__ = [
    'data_process',
    'build_features_for_bert',
]


def data_process(file_path: Union[str, List[str]], sep='\t', outer_label='O'):
    """
    数据示例：
    ```
    美	B-LOC
    国	I-LOC
    的	O
    华	B-PER
    莱	I-PER
    士	I-PER

    1
    9
    9
    3
    年
    ，
    我
    出
    生
    在
    北	B-LOC
    京	I-LOC
    ```

    Returns:
        [
            (['美', '国', '的', '华', '莱', '士'], ['B-LOC', 'I-LOC', 'O', 'B-PER', 'I-PER', 'I-PER']),
            (['1', '9', '9', '3', '年', '，', '我', '出', '生', '在', '北', '京'], ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-LOC', 'I-LOC']),
            ...
        ],
        { 'B-LOC', 'I-LOC', 'O', ... }

    """
    if isinstance(file_path, str):
        file_paths = [file_path]
    else:
        file_paths = file_path

    label_set = OrderedSet()
    label_set.add(outer_label)  # 确保 outer_label 在第一个，对应的 label_id 是 0

    rows = []
    for file_path in file_paths:
        with open(file_path, encoding='utf8') as f:
            tokens = []
            labels = []
            for ln in f:
                ln = ln.strip()
                if ln == '':  # next sample
                    if tokens:
                        rows.append((tokens, labels))
                        tokens = []
                        labels = []
                else:
                    tmp = ln.split(sep)
                    tokens.append(tmp[0])
                    if len(tmp) > 1:
                        labels.append(tmp[1])
                        label_set.add(tmp[1])
                    else:
                        labels.append(outer_label)

            if tokens:
                rows.append((tokens, labels))

    return rows, label_set


def build_features_for_bert(file_path: Union[str, List[str]],
                            sep='\t',
                            outer_label='O',
                            max_len=128,
                            tokenizer=None,
                            n_special_tokens=2,  # [CLS] and [SEP]
                            print_top_n=3):
    """"""
    if tokenizer is None:
        from my.nlp.bert_tokenizer import tokenizer

    data, label_types = data_process(file_path, sep=sep, outer_label=outer_label)
    label_map = {label: i for i, label in enumerate(label_types)}

    features = []
    for idx, (tokens, labels) in enumerate(data):
        txt = ' '.join(tokens)
        # tokens_len = len(tokens)
        # 用于计算 batch 内的 max_len，然后再截断（看到有人这么做，不确定在有 mask 的情况下是否需要）
        # 决定移除，有需要的话可以利用 mask 推断
        token_ids, token_type_ids, masks = tokenizer.encode(txt, max_len=max_len)

        # padding labels to max_len
        if len(tokens) > max_len - n_special_tokens:
            labels = [outer_label] + labels[: max_len - n_special_tokens] + [outer_label]
        else:
            labels = [outer_label] + labels + [outer_label] \
                     + [outer_label] * (max_len - n_special_tokens - len(labels))

        # for [CLS] and [SEP]
        label_ids = [label_map[x] for x in labels]

        assert max_len == len(token_ids) == len(token_type_ids) == len(masks) == len(label_ids)
        features.append([token_ids, token_type_ids, masks, label_ids])

        if idx < print_top_n:
            logger.info("*** Example %s ***" % idx)
            logger.info("\ttokens: %r", tokens)
            logger.info("\ttoken_ids: %r", token_ids)
            logger.info("\ttoken_type_ids: %r", token_type_ids)
            logger.info("\tmask: %r", masks)
            logger.info("\tlabel_ids: %r", label_ids)

    assert len(data) == len(features)
    return data, features, label_map


def _test():
    """"""
    test_file = os.path.join(os.path.dirname(__file__), 'test_data/ner_test_data.txt')

    def _test_data_process():
        """"""
        ds = data_process(test_file)

    _test_data_process()
    print()

    def _test_build_features_for_bert():
        """"""
        build_features_for_bert(test_file, max_len=16, print_top_n=2)

    _test_build_features_for_bert()


if __name__ == '__main__':
    """"""
    _test()
