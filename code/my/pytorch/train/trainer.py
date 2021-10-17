#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time: 2021-07-16 1:21 下午

Author: huayang

Subject:

"""
import math

from typing import *

import torch
import torch.nn as nn

try:
    accelerate_available = True
    from accelerate import Accelerator
except:  # noqa
    accelerate_available = False
    Accelerator = Any

from tqdm import tqdm
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from my.python.utils import get_logger, set_default, set_attr, get_print_json
from my.pytorch.train.config_utils import DEFAULT_ARGS, ARGS_TYPE
from my.pytorch.train.callback import Callback, ProgressbarCallback, ModelSaveCallback
from my.pytorch.train.optimizer import get_opt_by_name
from my.pytorch.train.scheduler import get_linear_schedule_with_warmup

# from my.pytorch.train.accelerator import SimpleAccelerator

logger = get_logger()

# Alias Type
Batch = Any
Loss = torch.Tensor

__all__ = [
    'Trainer'
]


class Trainer:
    """@Pytorch Utils
    一个简单的 Pytorch Trainer

    Examples:
        >>> # 参考 code/examples/pytorch/train_ner_bert_crf.py
    """
    args: ARGS_TYPE
    model: nn.Module
    params: Iterable[Union[torch.Tensor, Dict[str, Any]]]
    optimizer: Optimizer
    scheduler: Union[LambdaLR, Any]
    callbacks: List[Callback]
    forward_wrap: Callable[[nn.Module, Batch], Loss]

    current_batches: tqdm
    current_epoch: int
    current_batch: Union[List, Dict, Any]
    current_batch_idx: int
    current_batch_loss: torch.Tensor
    global_step: int
    # batch_step_count: int = 0

    stop_training: bool = False
    updating_gradient: bool = False

    def __init__(self,
                 args: ARGS_TYPE,
                 model: nn.Module,
                 data_train: DataLoader,
                 data_val: DataLoader = None,
                 forward_wrap: Callable[[nn.Module, Batch], Loss] = None,
                 callbacks: List[Type[Callback]] = None,
                 accelerator: "Accelerator" = None):
        """"""
        self.args = args
        self.model = model
        self.data_train = data_train
        self.data_val = data_val

        if forward_wrap is not None:
            self.forward_wrap = forward_wrap
        else:
            def forward_wrap(_model, _batch):
                return _model(**_batch)

            self.forward_wrap = forward_wrap

        self._callbacks = callbacks

        if accelerate_available and accelerator is None:
            accelerator = Accelerator()
            device = accelerator.device
            set_attr(args, 'device', device.type)
        else:
            device = set_default(args, 'device', DEFAULT_ARGS.device)
            # accelerator = SimpleAccelerator(device)
        self._accelerator = accelerator

        # 设置参数默认值
        self.device = device
        self.learning_rate = set_default(args, 'learning_rate', DEFAULT_ARGS.learning_rate)
        self.weight_decay = set_default(args, 'weight_decay', DEFAULT_ARGS.weight_decay)
        self.num_gpu = set_default(args, 'num_gpu', DEFAULT_ARGS.num_gpu)
        self.num_gradient_accumulation = set_default(args, 'num_gradient_accumulation',
                                                     DEFAULT_ARGS.num_gradient_accumulation)
        self.num_train_epochs = set_default(args, 'num_train_epochs', DEFAULT_ARGS.num_train_epochs)
        self.num_train_steps = set_default(args, 'num_train_steps', DEFAULT_ARGS.num_train_steps)
        if self.num_train_steps < 0:  # default -1
            self.num_train_steps = self.num_train_epochs * math.ceil(
                len(self.data_train) / self.num_gradient_accumulation)
            set_attr(args, 'num_train_steps', self.num_train_steps)

        self.val_per_step = set_default(args, 'val_per_step', DEFAULT_ARGS.val_per_step)
        if self.val_per_step < 0:  # default -1
            self.val_per_step = self.num_train_steps // 10
            set_attr(args, 'val_per_step', self.val_per_step)

        self.num_warmup_steps = set_default(args, 'num_warmup_steps', DEFAULT_ARGS.num_warmup_steps)
        self.optimizer_name = set_default(args, 'optimizer_name', DEFAULT_ARGS.optimizer_name)
        self.no_decay_params = set_default(args, 'no_decay_params', DEFAULT_ARGS.no_decay_params)
        self.save_state_dict = set_default(args, 'save_state_dict', DEFAULT_ARGS.save_state_dict)
        self.save_old_format = set_default(args, 'save_old_format', DEFAULT_ARGS.save_old_format)
        self.save_dir = set_default(args, 'save_dir', DEFAULT_ARGS.save_dir)
        self._global_step: list = set_default(args, 'global_step', DEFAULT_ARGS.global_step)

        # show args
        logger.info(get_print_json(args))

    @property
    def global_step(self):
        return self._global_step[-1]

    @global_step.setter
    def global_step(self, value):
        self._global_step[-1] = value

    @property
    def loss_item(self):
        """"""
        try:
            return self.current_batch_loss.item()
        except:
            return float('nan')

    def train(self):
        """"""
        self._on_before_train()

        for self.current_epoch in range(self.num_train_epochs):
            self._on_before_train_epoch()

            for self.current_batch_idx, self.current_batch in enumerate(self.current_batches):
                self._on_before_train_batch()

                if self.stop_training:
                    break

                self.current_batch_loss = self._compute_loss()
                self._loss_backward(self.current_batch_loss)

                # update gradient
                if self.updating_gradient:
                    self._on_before_optimizer_step()

                    self.global_step += 1
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    # 注意顺序: loss.backward() -> optimizer.step() -> scheduler.step() -> optimizer.zero_grad()

                    self._on_after_optimizer_step()

                self._on_after_train_batch()

            self._on_after_train_epoch()

        self._on_after_train()

    def _on_before_train(self):
        """"""
        # init
        self.callbacks = self._init_callbacks(self._callbacks)
        self.params = self._prepare_model_params(self.model)
        self.optimizer = self._init_optimizer(self.params)
        self.scheduler = self._init_scheduler(self.optimizer)

        # self.model, self.data_train, self.optimizer = self._accelerator.prepare(
        #     self.model, self.data_train, self.optimizer
        # )

        if self._accelerator is not None:
            self.model, self.data_train, self.optimizer = self._accelerator.prepare(
                self.model, self.data_train, self.optimizer
            )
        else:
            self.model = self.model.to(self.device)

        self.model.zero_grad()

        # invoke callbacks
        for callback in self.callbacks:
            callback.on_before_train()

    def _on_after_train(self):
        """"""
        # invoke callbacks
        for callback in self.callbacks:
            callback.on_after_train()

    def _on_before_train_epoch(self):
        """"""
        self.model.train()
        self.current_batches = tqdm(self.data_train)

        # invoke callbacks
        for callback in self.callbacks:
            callback.on_before_train_epoch()

    def _on_after_train_epoch(self):
        """"""
        self.current_batches.close()

        # invoke callbacks
        for callback in self.callbacks:
            callback.on_after_train_epoch()

    def _on_before_train_batch(self):
        """"""
        self.stop_training = self.global_step >= self.num_train_steps
        self.updating_gradient = (self.current_batch_idx + 1) % self.num_gradient_accumulation == 0 or (
                self.current_batch_idx + 1) == len(self.data_train)

        # invoke callbacks
        for callback in self.callbacks:
            callback.on_before_train_batch()

    def _on_after_train_batch(self):
        """"""
        # invoke callbacks
        for callback in self.callbacks:
            callback.on_after_train_batch()

    def _on_before_optimizer_step(self):
        """"""
        for callback in self.callbacks:
            callback.on_before_optimizer_step()

    def _on_after_optimizer_step(self):
        """"""
        for callback in self.callbacks:
            callback.on_after_optimizer_step()

    def _prepare_model_params(self, model: nn.Module) -> Iterable[Union[torch.Tensor, Dict[str, Any]]]:
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

    def _init_optimizer(self, params: Iterable) -> Optimizer:
        """ 设置优化器 """
        return get_opt_by_name(self.optimizer_name)(params)

    def _init_scheduler(self, optimizer):
        """"""
        return get_linear_schedule_with_warmup(optimizer, self.num_warmup_steps, self.num_train_steps)

    def _compute_loss(self):
        """"""
        try:
            loss = self.forward_wrap(self.model, self.current_batch)
        except:
            raise Exception(f'The model and batch not match, overwrite the {self.forward_wrap.__name__}')
        loss = loss.mean() / self.num_gradient_accumulation
        return loss

    def _loss_backward(self, loss):
        """"""
        # self._accelerator.backward(loss)
        if self._accelerator is not None:
            self._accelerator.backward(loss)
        else:
            loss.backward()

    def _init_callbacks(self, callbacks: List[Type[Callback]]):
        """"""

        # set callbacks
        if callbacks is not None:
            if not any(issubclass(callback, ProgressbarCallback) for callback in callbacks):
                callbacks.insert(0, ProgressbarCallback)
            if not any(issubclass(callback, ModelSaveCallback) for callback in callbacks):
                callbacks.append(ModelSaveCallback)
        else:
            callbacks = [ProgressbarCallback, ModelSaveCallback]

        # callbacks_init = []
        # for callback in callbacks:
        #     """"""
        #     if isinstance(callback, type):
        #         callback = callback(self)
        #     callbacks_init.append(callback)

        return [callback(self) for callback in callbacks]


def _test():
    """"""


if __name__ == '__main__':
    """"""
    _test()
