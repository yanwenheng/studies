#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time: 2021-08-17 11:22 上午

Author: huayang

Subject:

"""

import torch.nn as nn

from pytorch_models.modules import BertCRF
from my.pytorch.train.config_utils import TrainConfig
from my.pytorch.train.trainer import Trainer
from my.pytorch.train.callback import Callback
from my.pytorch.train.data_utils import NerBertDatasets
from my.python.utils import get_logger
from my.nlp.ner_utils import ner_result_parse

from itertools import islice
from my.nlp.bert_tokenizer import tokenizer

logger = get_logger(__name__)


def encode_batch(model: nn.Module, batch):
    token_ids, token_type_ids, masks, label_ids = batch
    _, loss = model([token_ids, token_type_ids, masks], label_ids, masks)
    return loss


class ExampleCallback(Callback):
    """"""

    def on_update_gradient_end(self):
        T = self.trainer

        if not T.global_step % 5 == 0:
            return

        model = T.model
        batch = T.current_batch

        token_ids, token_type_ids, masks, label_ids = batch
        prob, _ = model([token_ids, token_type_ids, masks], label_ids, masks)
        token_ids, mask = batch[0], batch[2]
        tags = model.decode(prob, mask)
        tags = tags.squeeze(0).cpu().numpy().tolist()
        print()
        for tokens_id, m, ids in islice(zip(token_ids, mask, tags), 5):
            tokens_id = tokens_id[m.to(bool)].cpu().numpy().tolist()  # 移除 [PAD]
            ids = ids[: len(tokens_id)]
            tokens_id = tokens_id[1: -1]  # 移除 [CLS]、[SEP]
            ids = ids[1: -1]
            chunks = ner_result_parse(tokens_id, ids,
                                      token_id2name=tokenizer.id2token_map,
                                      label_id2name=T.args.id2label_map)
            tokens = tokenizer.convert_ids_to_tokens(tokens_id)
            print(''.join(tokens), chunks)


# class train(Trainer):  # noqa
#     """"""
#
#     def _encode_batch(self, model: nn.Module, batch):
#         token_ids, token_type_ids, masks, label_ids = batch
#         probs, loss = model([token_ids, token_type_ids, masks], label_ids, masks)
#         return probs, loss
#
#     def run_on_step(self, global_step):
#         """"""
#         if not global_step % 5 == 0:
#             return
#
#         model = self.model
#         batch = self.current_batch
#
#         token_ids, token_type_ids, masks, label_ids = batch
#         prob, _ = model([token_ids, token_type_ids, masks], label_ids, masks)
#         token_ids, mask = batch[0], batch[2]
#         tags = model.decode(prob, mask)
#         tags = tags.squeeze(0).cpu().numpy().tolist()
#         print()
#         for tokens_id, m, ids in islice(zip(token_ids, mask, tags), 5):
#             tokens_id = tokens_id[m.to(bool)].cpu().numpy().tolist()  # 移除 [PAD]
#             ids = ids[: len(tokens_id)]
#             tokens_id = tokens_id[1: -1]  # 移除 [CLS]、[SEP]
#             ids = ids[1: -1]
#             chunks = ner_result_parse(tokens_id, ids,
#                                       token_id2name=tokenizer.id2token_map,
#                                       label_id2name=self.args.id2label_map)
#             tokens = tokenizer.convert_ids_to_tokens(tokens_id)
#             print(''.join(tokens), chunks)
#
#     def _parse_outputs(self, model, batch):
#         """"""
#         from itertools import islice
#         from my.nlp.bert_tokenizer import tokenizer
#
#         if self.global_step % 5 == 0:
#             token_ids, token_type_ids, masks, label_ids = batch
#             prob, _ = model([token_ids, token_type_ids, masks], label_ids, masks)
#             token_ids, mask = batch[0], batch[2]
#             tags = model.decode(prob, mask)
#             tags = tags.squeeze(0).cpu().numpy().tolist()
#             print()
#             for tokens_id, m, ids in islice(zip(token_ids, mask, tags), 5):
#                 tokens_id = tokens_id[m.to(bool)].cpu().numpy().tolist()  # 移除 [PAD]
#                 ids = ids[: len(tokens_id)]
#                 tokens_id = tokens_id[1: -1]  # 移除 [CLS]、[SEP]
#                 ids = ids[1: -1]
#                 chunks = ner_result_parse(tokens_id, ids,
#                                           token_id2name=tokenizer.id2token_map,
#                                           label_id2name=self.args.id2label_map)
#                 tokens = tokenizer.convert_ids_to_tokens(tokens_id)
#                 print(''.join(tokens), chunks)


def _test():
    """"""
    args = TrainConfig(src_train=r'data/train_data_100.txt', batch_size=8, val_percent=0, max_len=24)
    # args.num_gradient_accumulation = 1
    # args.num_train_steps = 20

    data = NerBertDatasets(args)
    args.id2label_map = data.id2label_map
    logger.info(data.id2label_map)

    model = BertCRF(n_classes=len(data.label_set))
    # args_dt = args.to_dict()
    # print(Namespace(**args))

    # args.num_train_steps = 5
    trainer = Trainer(args, model, data.train_set, data_val=data.val_set,
                      forward_wrap=encode_batch,
                      callbacks=[ExampleCallback])
    # trainer._encode_batch = encode_batch
    trainer.training()
    # train(model, data, args_dt)
    # train(model, data, args_ns)


if __name__ == '__main__':
    """"""

    _test()
