#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time: 2021-08-17 11:22 上午

Author: huayang

Subject:

"""
from my.pytorch.modules import BertCRF
from my.pytorch.pipeline import TrainConfig, NerBertDatasets
from my.pytorch.pipeline.trainer import Trainer
from my.python.utils import get_logger
from my.nlp.ner_utils import parse_labels

logger = get_logger(__name__)


class train(Trainer):
    """"""

    def _encode_batch(self, batch):
        """"""
        token_ids, token_type_ids, masks, label_ids = batch
        outputs = self.model([token_ids, token_type_ids, masks], label_ids, masks)
        return outputs

    def _parse_outputs(self):
        """"""
        from itertools import islice
        from my.nlp.bert_tokenizer import tokenizer

        if self.global_step[-1] % 5 == 0:
            prob, _ = self._batch_outputs
            token_ids, mask = self._batch[0], self._batch[2]
            tags = self.model.decode(prob, mask)
            # print(f'tags_shape: {tags.shape}, tags: {tags}')
            tags = tags.squeeze(0).cpu().numpy().tolist()
            print()
            for tokens_id, m, ids in islice(zip(token_ids, mask, tags), 5):
                tokens_id = tokens_id[m.to(bool)].cpu().numpy().tolist()  # 移除 [PAD]
                ids = ids[: len(tokens_id)]  # [1: -1]  # 移除 [CLS]、[SEP]
                tokens_id = tokens_id  # [1: -1]
                chunks = parse_labels(tokens_id, ids, token_id2name=tokenizer.id2token_map,
                                      label_id2name=self._args.id2label_map)
                tokens = tokenizer.convert_ids_to_tokens(tokens_id)
                print(''.join(tokens), chunks)


def _test():
    """"""
    from argparse import Namespace
    args = TrainConfig(src_train=r'data/train_data_100.txt', batch_size=8, val_percent=0, max_len=24)
    args.num_train_steps = 20

    data = NerBertDatasets(args)
    args.id2label_map = data.id2label_map
    logger.info(args.id2label_map)

    model = BertCRF(n_classes=len(data.label_set))
    args_dt = args.to_dict()
    args_ns = Namespace(**args_dt)

    train(model, data, args)
    # train(model, data, args_dt)
    # train(model, data, args_ns)


if __name__ == '__main__':
    """"""

    _test()
