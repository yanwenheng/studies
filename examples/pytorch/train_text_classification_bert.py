#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time: 2021-08-17 11:22 上午

Author: huayang

Subject:

"""
import os
import sys
import json
import platform

try:
    flag = True
    if len(sys.argv) > 1:
        pkg_path, ckpt_path = sys.argv[1], sys.argv[2]
    elif platform.system() == 'Linux':
        # 默认服务器实验环境
        pkg_path = r'/home/hadoop-aipnlp/cephfs/data/huayang04/lab'
        ckpt_path = r'/home/hadoop-aipnlp/cephfs/data/huayang04/ckpt'
    else:
        flag = False
        pkg_path = ''
        ckpt_path = ''

    if flag:
        sys.path.append(pkg_path)
        os.environ['CKPT'] = ckpt_path
except:  # noqa
    pass

from my.pytorch.modules.transformer import get_bert_pretrained, BertClassification
from my.pytorch.pipeline import TrainConfig, BertDatasets
from my.pytorch.pipeline.trainer import Trainer

from my.python.utils import get_logger
from my.python.custom_argparse import simple_argparse

logger = get_logger(__name__)


class TmpDatasets(BertDatasets):
    """

    ex: {"label": "102",
         "label_desc": "news_entertainment",
         "sentence": "江疏影甜甜圈自拍，迷之角度竟这么好看，美吸引一切事物",
         "keywords": "江疏影,美少女,经纪人,甜甜圈"}
    """

    def _process_line(self, line, src_type=None):
        """"""
        it = json.loads(line)
        if src_type != 'test':
            return [it['sentence'], it['label_desc']]
        else:
            return [it['sentence']]


class train(Trainer):
    """"""

    # def _encode_batch(self, batch):
    #     """"""
    #     token_ids, token_type_ids, masks, label_ids = batch
    #     outputs = self.model()
    #     return outputs

    # def _parse_outputs(self):
    #     """"""
    #     from itertools import islice
    #     from my.nlp.bert_tokenizer import tokenizer
    #
    #     if self.global_step[-1] % 5 == 0:
    #         prob, _ = self._batch_outputs
    #         token_ids, mask = self._batch[0], self._batch[2]
    #         tags = self.model.decode(prob, mask)
    #         # print(f'tags_shape: {tags.shape}, tags: {tags}')
    #         tags = tags.squeeze(0).cpu().numpy().tolist()
    #         print()
    #         for tokens_id, m, ids in islice(zip(token_ids, mask, tags), 5):
    #             tokens_id = tokens_id[m.to(bool)].cpu().numpy().tolist()  # 移除 [PAD]
    #             ids = ids[: len(tokens_id)]  # [1: -1]  # 移除 [CLS]、[SEP]
    #             tokens_id = tokens_id  # [1: -1]
    #             chunks = parse_labels(tokens_id, ids, token_id2name=tokenizer.id2token_map,
    #                                   label_id2name=self._args.id2label_map)
    #             tokens = tokenizer.convert_ids_to_tokens(tokens_id)
    #             print(''.join(tokens), chunks)


def _test():
    """"""
    dp = os.path.dirname(__file__)
    args = TrainConfig(src_train=os.path.join(dp, r'data/tnews_sample/train.json'),
                       src_val=os.path.join(dp, r'data/tnews_sample/dev.json'),
                       src_test=os.path.join(dp, r'data/tnews_sample/test.json'),
                       batch_size=32,
                       max_len=128)

    simple_argparse(args)
    print('num_gradient_accumulation:', args.num_gradient_accumulation)

    data = TmpDatasets(args, dict_batch=True, num_examples=1)
    args.id2label_map = data.id2label_map
    logger.info(args.id2label_map)

    bert = get_bert_pretrained()
    model = BertClassification(n_classes=len(args.id2label_map), bert=bert)
    train(model, data, args, show_example=False)


if __name__ == '__main__':
    """"""
    _test()
