#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time:
   2021-07-21 4:50 下午
   
Author:
   huayang
   
Subject:
   
"""

import torch
import torch.nn as nn
import torch.optim as optim  # noqa

from torch.optim import Optimizer

from typing import Dict

STR2OPT = {
    'Adam': optim.Adam,
    'AdamW': optim.AdamW,
    'SGD': optim.SGD,
}


def get_opt_by_name(opt_name: str):
    """"""
    if opt_name in STR2OPT:
        return STR2OPT[opt_name]

    try:
        return getattr(optim, opt_name)
    except:
        raise ValueError(f'No Optimizer named `{opt_name}` in `torch.optim`.')
