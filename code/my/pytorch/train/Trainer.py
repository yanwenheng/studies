#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time: 2021-07-16 1:21 下午

Author: huayang

Subject:

"""
import os
import math

from typing import List, Dict

import torch
import torch.nn as nn

try:
    accelerate_available = True
    from accelerate import Accelerator
except:  # noqa
    accelerate_available = False

from tqdm import tqdm
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from my.python.utils import get_logger, set_default, get_print_json
from my.pytorch.train.config import DEFAULT_ARGS, ARGS_TYPE
from my.pytorch.data_utils.Datasets import Datasets
from my.pytorch.train.optimizer import get_opt_by_name
from my.pytorch.train.scheduler import get_linear_schedule_with_warmup

logger = get_logger()


class Trainer:
    """"""

    def __init__(self,
                 model: nn.Module,
                 data: Datasets,
                 args: ARGS_TYPE,
                 accelerator=None,
                 show_example=True):
        """"""
        self.model = model
        self.data = data
        self._args = args

        # self._use_accelerator = accelerate_available and use_accelerator
        # if accelerate_available is False and use_accelerator is True:
        #     logger.warning('accelerate is not available, but `use_accelerator` is True.')

        self.device = set_default(args, 'device', DEFAULT_ARGS.device)

        if accelerate_available and accelerator is None:
            accelerator = Accelerator(cpu=(self.device == 'cpu'))
        self._accelerator = accelerator

        # 设置参数默认值
        self.learning_rate = set_default(args, 'learning_rate', DEFAULT_ARGS.learning_rate)
        self.weight_decay = set_default(args, 'weight_decay', DEFAULT_ARGS.weight_decay)
        self.num_gpu = set_default(args, 'num_gpu', DEFAULT_ARGS.num_gpu)
        self.num_train_epochs = set_default(args, 'num_train_epochs', DEFAULT_ARGS.num_train_epochs)
        self.num_train_steps = set_default(args, 'num_train_steps', DEFAULT_ARGS.num_train_steps)
        self.num_warmup_steps = set_default(args, 'num_warmup_steps', DEFAULT_ARGS.num_warmup_steps)
        self.num_gradient_accumulation = set_default(args, 'num_gradient_accumulation',
                                                     DEFAULT_ARGS.num_gradient_accumulation)
        self.optimizer_name = set_default(args, 'optimizer_name', DEFAULT_ARGS.optimizer_name)
        self.no_decay_params = set_default(args, 'no_decay_params', DEFAULT_ARGS.no_decay_params)
        self.global_step = set_default(args, 'global_step', DEFAULT_ARGS.global_step)
        self.save_state_dict = set_default(args, 'save_state_dict', DEFAULT_ARGS.save_state_dict)
        self.save_old_format = set_default(args, 'save_old_format', DEFAULT_ARGS.save_old_format)
        self.save_dir = set_default(args, 'save_dir', DEFAULT_ARGS.save_dir)

        if self.num_train_steps < 0:  # default -1
            self.num_train_steps = self.num_train_epochs * math.ceil(
                len(self.data.train_set) / self.num_gradient_accumulation)

        # 没有完全使用成员变量，而是采用传参的方式，是为了更好的体现依赖，也方便子类重写方法
        self.params = self._init_training_params(self.model)
        self.optimizer = self._init_optimizer(self.params)
        self.scheduler = self._init_scheduler(self.optimizer)
        self._w_epoch = len(str(self.num_train_epochs))  # epoch 显示宽度
        self._w_step = len(str(self.num_train_steps))  # step 显示宽度
        self._show_example = show_example

        if self._accelerator is not None:
            args.device = self._accelerator.device.type
            self.model, self.data.train_set, self.optimizer = accelerator.prepare(
                self.model, self.data.train_set, self.optimizer
            )
        else:
            self.model = self.model.to(self.device)

        # show args
        logger.info(get_print_json(args))
        # start training
        self._training()
        # save the model and config
        self._save_model()

    def _init_training_params(self, model) -> List[Dict]:
        """"""
        named_parameters = list(model.named_parameters())
        params = [
            {
                'params': [p for n, p in named_parameters if not any(nd in n for nd in self.no_decay_params)],
                'weight_decay': self.weight_decay,
                'lr': self.learning_rate
            },
            {
                'params': [p for n, p in named_parameters if any(nd in n for nd in self.no_decay_params)],
                'weight_decay': 0.0,
                'lr': self.learning_rate
            }
        ]

        return params

    def _init_optimizer(self, params) -> Optimizer:
        """ 设置优化器 """
        return get_opt_by_name(self.optimizer_name)(params)

    def _init_scheduler(self, optimizer) -> LambdaLR:
        """"""
        return get_linear_schedule_with_warmup(optimizer, self.num_warmup_steps, self.num_train_steps)

    def _training(self):
        """"""
        self.model.zero_grad()
        for epoch in range(self.num_train_epochs):
            self._epoch = epoch

            # train
            self.model.train()
            self._train_epoch()

            # TODO: eval_epoch()

    def _train_epoch(self):
        """ epoch train """
        self._batches = tqdm(self.data.train_set)
        self._set_progressbar_description()

        for step, batch in enumerate(self._batches):
            if self.global_step[-1] >= self.num_train_steps:
                break

            self._step, self._batch = step, batch
            self._train_step(step, batch)

        self._batches.close()

    def _train_step(self, step, batch):
        """ step train """
        self._batch_outputs = self._encode_batch(batch)
        self._loss = self._batch_outputs[-1]  # default
        self._loss = self._loss.mean() / self.num_gradient_accumulation
        self._loss_backward(self._loss)

        if (step + 1) % self.num_gradient_accumulation == 0 \
                or (step + 1) == len(self.data.train_set):
            self._update_step()

        self._set_progressbar_postfix()

    def _loss_backward(self, loss):
        """"""
        if self._accelerator is not None:
            self._accelerator.backward(loss)
        else:
            loss.backward()

    def _encode_batch(self, batch) -> torch.Tensor:
        """ 当 batch 与模型输入不是完全匹配时，可能需要重写本方法

        Examples:
            ```
            token_ids, token_type_ids, masks, labels = self.batch
            inputs = [token_ids, token_type_ids, masks]
            return self.model(inputs, labels=labels, masks=masks)
            ```
        """
        try:
            return self.model(**batch)  # 默认 batch 为字典格式
        except:
            raise NotImplementedError(f'It seems that batch not match the model, '
                                      f'overwrite the `{self._encode_batch.__name__}` function.')

    def _update_step(self):
        """ step 内需要更新的对象 """
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        self.global_step[-1] += 1

        self._set_progressbar_description()
        self._batches.update()  # 更新进度条

        if self._show_example:
            self.model.eval()
            with torch.no_grad():
                self._parse_outputs()
            self.model.train()

    def _set_progressbar_postfix(self):
        """ 在进度条中添加其他信息 """
        self._batches.set_postfix(loss=self._loss.item())

    def _set_progressbar_description(self):
        """ 更新进度条描述
        默认格式: Global Step[02/39] - Epoch(0):  23%|██▎       | 3/13 [00:05<00:16,  1.60s/it, loss=6.24]
        """
        self._batches.set_description(
            f'Global Step[{self.global_step[-1]:>0{self._w_step}}/{self.num_train_steps}] - '
            f'Epoch({self._epoch:>0{self._w_epoch}})'
        )

    def _parse_outputs(self):
        """ 解析输出用于展示示例
        Examples:
            ```
            if self.global_step % 10 == 0:
                # parse the `self._batch_outputs`
            ```
        """
        if self._show_example:
            raise NotImplementedError(f'It need overwrite the `{self._parse_outputs.__name__}` function '
                                      f'when `_show_example` is True.')

    def _save_model(self):
        """"""
        # 保存模型
        save_obj = self.model.state_dict() if self.save_state_dict else self.model
        model_save_path = os.path.join(self.save_dir, 'model.pt')
        os.makedirs(self.save_dir, exist_ok=True)
        torch.save(save_obj, model_save_path, _use_new_zipfile_serialization=not self.save_old_format)

        # 保存配置
        config_save_path = os.path.join(self.save_dir, 'train_config.json')
        with open(config_save_path, 'w', encoding='utf8') as fw:
            fw.write(get_print_json(self._args))


def _test():
    """"""


if __name__ == '__main__':
    """"""
    _test()
