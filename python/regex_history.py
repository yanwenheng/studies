#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time: 2021-10-05 11:47 下午

Author: huayang

Subject:

"""
import re
import os
import sys
import json
import doctest

from typing import *
from collections import defaultdict


RE_MULTI_LINE = re.compile(r'(\n\s*)+')
"""匹配多个换行符"""


def _test():
    """"""
    doctest.testmod()


if __name__ == '__main__':
    """"""
    _test()
