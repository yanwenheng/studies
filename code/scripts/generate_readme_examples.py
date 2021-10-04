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
    script_path = os.path.dirname(__file__)
    code_path = os.path.join(script_path, '..')
    repo_path = os.path.join(code_path, '..')
    # print(os.path.basename(os.path.abspath(repo_path)))
    sys.path.append(code_path)
except:  # noqa
    pass

from my.python.code_analysis import DocParser, module_iter
from my.python.file_utils import files_concat
from my.python.custom import simple_argparse


def hn_line(line, lv=2):
    """"""
    return f'{"#" * lv} {line}'


def gen_readme(args, docs_dt: Dict[str, List[DocParser]]):
    """"""
    code_prefix = os.path.basename(os.path.abspath(code_path))

    toc = [args.toc_line, '---\n']
    # main_toc = ['My Code Lab', '---\n']
    content = []
    append_content = []

    beg_details_tmp = '<details><summary><b> {key} <a href="{url}">¶</a></b></summary>\n'
    end_details = '\n</details>\n'

    key_sorted = sorted(docs_dt.keys())
    for key in key_sorted:
        blocks = docs_dt[key]
        toc.append(beg_details_tmp.format(key=key, url=f'#{DocParser.slugify(key)}'))
        # main_toc.append(beg_details_tmp.format(key=key, url=f'{code_prefix}/README.md#{DocParser.slugify(key)}'))

        content.append(hn_line(key, 2))
        append_content.append(hn_line(key, 2))
        for d in blocks:
            toc.append('- ' + DocParser.get_toc_line(d.summary_line))
            # main_toc.append('- ' + DocParser.get_toc_line(d.summary_line, prefix=code_prefix))
            content.append(d.get_markdown_block())
            append_content.append(d.get_markdown_block(prefix=code_prefix))

        toc.append(end_details)
        # main_toc.append(end_details)

    toc_str = '\n'.join(toc)
    sep = '\n---\n\n'
    content_str = '\n\n'.join(content)
    code_readme = toc_str + sep + content_str
    with open(args.out, 'w', encoding='utf8') as fw:
        fw.write(code_readme)

    main_append = toc_str + sep + '\n\n'.join(append_content)
    return main_append


def gen_code_readme(args):
    """ 生成 readme for code """
    toc_line = 'My Code Lab'
    args.toc_line = toc_line

    docs_dt = defaultdict(list)

    for module in module_iter(args.module_path):
        if hasattr(module, '__all__'):
            for obj_str in module.__all__:
                obj = getattr(module, obj_str)
                if isinstance(obj, (ModuleType, FunctionType, type)) \
                        and getattr(obj, '__doc__') \
                        and obj.__doc__.startswith('@'):
                    doc = DocParser(obj)
                    docs_dt[doc.flag[1:]].append(doc)

    code_append = gen_readme(args, docs_dt)
    return toc_line, code_append


def gen_main_toc(toc_lines):
    """"""
    lns = ['Study Index', '---\n']
    for ln in toc_lines:
        lns.append('- ' + DocParser.get_toc_line(ln))

    return lns


def pipeline():
    """"""
    args = simple_argparse()
    repo_readme_path = os.path.join(repo_path, r'README.md')
    if os.path.exists(repo_readme_path):
        readme_old = open(repo_readme_path, encoding='utf8').read()
    else:
        readme_old = ''
    code_toc, code_append = gen_code_readme(args)
    block_toc = gen_main_toc([code_toc])
    readme_main_path = os.path.join(repo_path, r'README-main.md')

    auto_line = '<font color="LightGrey"><i> `The following is Auto-generated` </i></font>'
    files_concat(src_in=[readme_main_path,
                         auto_line,
                         '\n'.join(block_toc),
                         code_append],
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
        command = "generate_readme_examples.py --module_path .. --out ../README.md"
        sys.argv = command.split()
        _test()
