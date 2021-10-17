#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time: 2021-10-16 12:56 下午

Author: huayang

Subject:

"""
import os
import abc
import sys
import json
import doctest

from typing import *
from collections import defaultdict

import torch

from my.pytorch import train
from my.python.utils import get_print_json, get_logger

logger = get_logger(__name__)


class Callback(abc.ABC):
    """"""
    trainer: "train.Trainer"  # 避免循环引用

    def __init__(self, trainer):
        # self(trainer)
        self.trainer = trainer

    def __call__(self, trainer):
        self.__init__(trainer)

    def on_before_train(self):
        """"""

    def on_after_train(self):
        """"""

    def on_before_train_epoch(self):
        """"""

    def on_after_train_epoch(self):
        """"""

    def on_before_train_batch(self):
        """"""

    def on_after_train_batch(self):
        """"""

    def on_before_optimizer_step(self):
        """"""

    def on_after_optimizer_step(self):
        """"""

    def on_before_eval(self):
        """"""

    def on_after_eval(self):
        """"""

    def on_before_test(self):
        """"""

    def on_after_test(self):
        """"""


class ProgressbarCallback(Callback):
    """"""

    def __init__(self, trainer):
        """"""
        super().__init__(trainer)
        self._w_epoch = len(str(trainer.num_train_epochs))  # epoch 显示宽度
        self._w_step = len(str(trainer.num_train_steps))  # step 显示宽度

    def on_after_train_batch(self):
        """"""
        self._set_progressbar_postfix()

    def on_before_train_epoch(self):
        """"""
        self._set_progressbar_postfix()
        self._set_progressbar_description()

    def on_after_optimizer_step(self):
        """"""
        self._set_progressbar_description()

    def _set_progressbar_postfix(self):  # noqa
        """ 在进度条中添加其他信息 """
        trainer = self.trainer
        trainer.current_batches.set_postfix(loss=trainer.loss_item)

    def _set_progressbar_description(self):
        """ 更新进度条描述
        默认格式: Global Step[02/39] - Epoch(0):  23%|██▎       | 3/13 [00:05<00:16,  1.60s/it, loss=6.24]
        """
        trainer = self.trainer
        trainer.current_batches.set_description(
            f'Global Step[{trainer.global_step:>0{self._w_step}}/{trainer.num_train_steps}] - '
            f'Epoch({trainer.current_epoch:>0{self._w_epoch}})'
        )


class ModelSaveCallback(Callback):
    """"""

    def __init__(self, trainer):
        super().__init__(trainer)

    def on_after_train(self):
        """"""
        self._save_model()

    def _save_model(self):
        """"""
        trainer = self.trainer

        # 保存模型
        save_obj = trainer.model.state_dict() if trainer.save_state_dict else trainer.model
        model_save_path = os.path.join(trainer.save_dir, 'model.pt')
        os.makedirs(trainer.save_dir, exist_ok=True)
        torch.save(save_obj, model_save_path, _use_new_zipfile_serialization=not trainer.save_old_format)

        # 保存配置
        config_save_path = os.path.join(trainer.save_dir, 'train_config.json')
        with open(config_save_path, 'w', encoding='utf8') as fw:
            fw.write(get_print_json(trainer.args))

        logger.info(f'model saved at {model_save_path}')


def _test():
    """"""
    doctest.testmod()


if __name__ == '__main__':
    """"""
    _test()
