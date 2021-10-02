#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time: 2021-09-18 3:31 下午

Author: huayang

Subject:

"""
import doctest

from typing import Dict

import numpy as np

__all__ = [
    'ner_result_parse'
]


def ner_result_parse(tokens, labels,
                     label_id2name: Dict[int, str],
                     token_id2name: Dict[int, str] = None):
    """@NLP Utils
    NER 结果解析（基于 BIO 格式）

    Examples:
        >>> _label_id2name = {0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-LOC', 4: 'I-LOC'}
        >>> _tokens = list('你知道小明生活在北京吗？')
        >>> _labels = list(map(int, '000120003400'))
        >>> ner_result_parse(_tokens, _labels, _label_id2name)
        [['PER', '小明', (3, 4)], ['LOC', '北京', (8, 9)]]

        >>> _tokens = list('小明生活在北京')  # 测试头尾是否正常
        >>> _labels = list(map(int, '1200034'))
        >>> ner_result_parse(_tokens, _labels, label_id2name=_label_id2name)
        [['PER', '小明', (0, 1)], ['LOC', '北京', (5, 6)]]

        >>> _tokens = list('明生活在北京')  # 明: I-PER
        >>> _labels = list(map(int, '200034'))
        >>> ner_result_parse(_tokens, _labels, label_id2name=_label_id2name)
        [['LOC', '北京', (4, 5)]]

        >>> _tokens = list('小明生活在北')
        >>> _labels = list(map(int, '120003'))  # 北: B-LOC
        >>> ner_result_parse(_tokens, _labels, label_id2name=_label_id2name)
        [['PER', '小明', (0, 1)], ['LOC', '北', (5, 5)]]

    Args:
        tokens:
        labels:
        token_id2name:
        label_id2name:

    Returns:
        example: [['小明', 'PER', (3, 4)], ['北京', 'LOC', (8, 9)]]
    """
    INIT_IDX = -1

    def _init():
        return '', INIT_IDX, INIT_IDX

    def get_tag():
        try:
            return label.split('-')[1]
        except:  # noqa
            return '_SPAN'  # 针对 'B'/'I' 而非 'B-XX'/'I-XX' 的情况

    def chunks_append():
        span = ''.join(tokens[beg: end + 1])
        chunks.append([tag, span, (beg, end)])

    # if masks is not None:
    #     tokens = np.asarray(tokens)[np.asarray(masks, dtype=bool)].tolist()
    #     labels = np.asarray(labels)[np.asarray(masks, dtype=bool)].tolist()

    if token_id2name is not None:
        tokens = [token_id2name.get(t, t) for t in tokens]
    if label_id2name is not None:
        labels = [label_id2name.get(t, t) for t in labels]

    assert len(tokens) == len(labels)
    SEQ_LEN = len(tokens) - 1

    chunks = []
    tag, beg, end = _init()
    for idx, (token, label) in enumerate(zip(tokens, labels)):
        if label.startswith('B'):
            if end != INIT_IDX:
                chunks_append()
            tag = get_tag()
            beg = end = idx

            if end == SEQ_LEN:
                chunks_append()
        elif label.startswith('I') and beg != INIT_IDX:
            _tag = get_tag()
            if _tag == tag:
                end = idx

            if end == SEQ_LEN:
                chunks_append()
        else:
            if end != INIT_IDX:
                chunks_append()
            tag, beg, end = _init()

    return chunks


def _test():
    """"""
    doctest.testmod()


if __name__ == '__main__':
    """"""
    _test()
