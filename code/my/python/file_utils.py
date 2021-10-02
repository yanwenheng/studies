#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time: 2021-10-01 10:50 下午

Author: huayang

Subject:

"""
import os
import json
import doctest

from typing import *
from collections import defaultdict


def files_concat(files_in: List[str], file_out: str, sep: str = ''):
    """@Python Utils
    文件拼接

    Examples:
        >>> _dir = r'./-test'
        >>> os.makedirs(_dir, exist_ok=True)
        >>> f1 = os.path.join(_dir, r't1.txt')
        >>> f2 = os.path.join(_dir, r't2.txt')
        >>> os.system(f'echo 123 > {f1}')
        0
        >>> os.system(f'echo 456 > {f2}')
        0
        >>> f_out = os.path.join(_dir, r't.txt')
        >>> files_concat([f1, f2], f_out)
        >>> print(open(f_out).read())
        123
        456
        <BLANKLINE>
        >>> files_concat([f1, f2], f_out, '---')
        >>> print(open(f_out).read())
        123
        ---
        456
        <BLANKLINE>
        >>> os.system(f'rm -rf {_dir}')
        0

    """

    def _fc(fp):
        fc = open(fp).read()
        if fc.endswith('\n'):
            return fc
        else:
            return fc + '\n'

    if sep and not sep.endswith('\n'):
        sep += '\n'

    buf = sep.join([_fc(fp) for fp in files_in])

    with open(file_out, 'w') as fw:
        fw.write(buf)


def _test():
    """"""
    doctest.testmod()


if __name__ == '__main__':
    """"""
    _test()
