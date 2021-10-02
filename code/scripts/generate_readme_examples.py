#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time: 2021-10-01 11:13 下午

Author: huayang

Subject:

"""
import os
import sys
import json
import doctest

from types import *
from typing import *
from collections import defaultdict
from pkgutil import walk_packages

from sortedcontainers import SortedList

try:
    dir_path = os.path.dirname(__file__)
    sys.path.append(os.path.join(dir_path, r'..'))
except:  # noqa
    pass

from my.python.code_analysis import DocParser, module_iter
from my.python.file_utils import files_concat
from my.python.custom import simple_argparse


def gen_readme(docs):
    """"""
    toc = ['My Code Lab', '---\n']
    content = []
    for key, blocks in docs.items():
        toc.append(DocParser.get_toc_line(key))
        content.append(f'## {key}')
        for toc_line, markdown_block in blocks:
            toc.append('    ' + toc_line)
            content.append(markdown_block)

    toc_str = '\n'.join(toc)
    sep = '\n\n' + '---' + '\n\n'
    content_str = '\n\n'.join(content)

    return toc_str + sep + content_str


def gen_readme_examples(args):
    """"""
    docs = defaultdict(list)

    for module in module_iter(args.module_path):
        if hasattr(module, '__all__'):
            for obj_str in module.__all__:
                obj = getattr(module, obj_str)
                if getattr(obj, '__doc__', None) and obj.__doc__.startswith('@'):
                    doc = DocParser(obj)
                    docs[doc.flag[1:]].append((doc.toc_line, doc.markdown_block))

    readme_ex = gen_readme(docs)
    with open(args.out, 'w', encoding='utf8') as fw:
        fw.write(readme_ex)

    # return readme_ex


def pipeline():
    """"""
    args = simple_argparse()
    repo_readme_path = os.path.join(dir_path, r'../../README.md')
    if os.path.exists(repo_readme_path):
        readme_old = open(repo_readme_path, encoding='utf8').read()
    else:
        readme_old = ''
    gen_readme_examples(args)
    readme_main_path = os.path.join(dir_path, r'../../README-main.md')
    files_concat(files_in=[readme_main_path, args.out],
                 file_out=repo_readme_path,
                 sep='\n---\n\n')
    readme = open(repo_readme_path, encoding='utf8').read()
    if readme_old != readme:
        print('DIFF')
    else:
        print('NO-DIFF')


def _test():
    """"""
    doctest.testmod()
    pipeline()


if __name__ == '__main__':

    if len(sys.argv) > 1:
        pipeline()
        # print('SUCCESS')
    else:
        # 抑制标准输出，只打印 WARNING 信息
        # sys.stdout = open(os.devnull, 'w')
        command = "generate_readme_examples.py --module_path ../my --out ../README.md"
        sys.argv = command.split()
        _test()
