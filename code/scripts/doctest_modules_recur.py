#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time: 2021-09-27 3:44 下午

Author: huayang

Subject: 递归对指定模块（指定文件夹下的所有模块）进行文档测试

Examples:
    # >>> doctest_modules(r'../my/pytorch')
    # 0
    # >>> doctest_modules([r'../my/python', r'../my/nlp'])
    # 0
    #
    # # 命令行调用
    # >>> os.system('python doctest_modules_recur.py ../my')
    # 0

"""
import os
import sys
import doctest
import logging
import pkgutil
import importlib

from typing import *

# from my.python.utils import get_logger, set_stdout_null

# logger = logging.getLogger(__name__)


def doctest_modules(paths: Union[str, List[str]]):
    """"""
    if isinstance(paths, str):
        paths = [paths]

    num_failed = 0
    for p in paths:
        # 示例：path = '../my'
        path = os.path.abspath(p)  # /Users/huayang/workspace/my/studies/code/my
        base = os.path.basename(path)  # my
        dir_path = os.path.dirname(path)  # /Users/huayang/workspace/my/studies/code
        sys.path.append(dir_path)  # 添加到环境变量

        if os.path.isdir(path):
            tmp_failed = _doctest_module(path, base)
        else:
            file_name, ext = os.path.splitext(base)
            tmp_failed = _doctest_py(file_name)
        num_failed += tmp_failed

    return num_failed


def _doctest_py(module_name):
    """"""
    num_failed = 0
    module = importlib.import_module(module_name)
    if hasattr(module, 'doctest'):  # 如果该模块中使用了文档测试
        num_failed = doctest.testmod(module).failed
        if num_failed > 0:
            logging.warning(f'=== `{module.__name__}` doctest failed! ===')

    return num_failed


def _doctest_module(path, base):
    """"""
    num_failed = 0
    for p, module_name, is_pkg in pkgutil.walk_packages([path], base + '.'):
        # print(path, module_name, is_pkg)
        num_failed += _doctest_py(module_name)

    return num_failed


def _test():
    """"""
    doctest.testmod()


if __name__ == '__main__':
    """"""
    sys.argv = 'code/scripts/doctest_modules_recur.py /Users/huayang/workspace/my/studies/code/my'.split()
    if len(sys.argv) > 1:
        pkg_path = sys.argv[1:]
        failed = doctest_modules(pkg_path)
        print(failed)
    else:
        # 抑制标准输出，只打印 WARNING 信息
        sys.stdout = open(os.devnull, 'w')

        # assert 0 == doctest_modules(r'../my/pytorch')
        # assert 0 == doctest_modules([r'../my/python', r'../my/nlp'])
        assert 0 == os.system('python doctest_modules_recur.py ../my')
