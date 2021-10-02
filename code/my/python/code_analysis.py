#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time: 2021-09-30 11:58 上午

Author: huayang

Subject: 源码解析

"""
import os
import re
import sys
import json
import doctest
import inspect

from types import *
from typing import *
from collections import defaultdict
from dataclasses import dataclass, field, fields
from pkgutil import walk_packages
from markdown.extensions.toc import slugify_unicode as slugify

from my.python.utils import get_logger
from my.python.custom import simple_argparse, ConfigDict

logger = get_logger(__name__)

RE_INDENT = re.compile(r'^([ ]*)(?=\S)', re.MULTILINE)
RE_SPACE = re.compile(r'^\s+')

SHOW_FLAG = '@show'
SHIELD_FLAG = '@shield'

README_BLOCK_TEMPLATE = """### {summary}
{details}
```python
{examples}
```
"""


def module_iter(path='.') -> Iterable[ModuleType]:
    """
    Examples:
        >>> _path = r'../'
        >>> modules = module_iter(_path)
        >>> m = next(modules)  # noqa
        >>> type(m)
        <class 'module'>

    """
    assert os.path.isdir(path)
    path = os.path.abspath(path)
    base = os.path.basename(path)
    for finder, module_name, is_pkg in walk_packages([path], base + '.'):
        loader = finder.find_module(module_name)
        module = loader.load_module(module_name)
        yield module


def add_docflag_dn(flag: str):  # noqa
    """"""

    def docstring_decorator(fn):
        fn.__doc__ = '\n'.join([f'{flag}'] + fn.__doc__.split('\n'))
        return fn

    return docstring_decorator


def docstring_example(a):
    """@Test
    docstring 示例，基于 Google 风格（一行简述）

    一个 Google 风格的 docstring 示例（函数具体说明）
        ...

    Args:
        a:

    Examples:
        >>> 1
        1
        >>> raise Exception
        Traceback (most recent call last):
            ...
        Exception

    References:@shield
        https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html
    """


@dataclass()
class Item:
    """"""
    name: str
    flag: str = SHOW_FLAG
    lines: List[str] = None


class DocParser:
    """@Test
    一个简单的 docstring 解析器，只解析了我需要的部分（比如 Examples，用于自动生成 README）

    Examples:
        >>> from my.python.code_analysis import DocParser
        >>> def example_fn(a):
        ...     '''@show
        ...     docstring 示例，基于 Google 风格（一行简述）
        ...
        ...     一个 Google 风格的 docstring 示例（函数具体说明）
        ...         ...
        ...
        ...     Args:
        ...         a: arg a
        ...
        ...     Examples:
        ...         >>> 1
        ...         1
        ...
        ...     References:@shield
        ...         https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html
        ...     '''
        >>> dp = DocParser(example_fn)
        >>> dp.summary.lines
        ['`example_fn`: ', 'docstring 示例，基于 Google 风格（一行简述）']

    """
    _obj: Union[Callable, Type]

    def __init__(self, obj):
        """"""
        self.summary: Item = Item(name='Summary')  # 概述：对应 docstring 的第一行
        self.details: Item = Item(name='Details')  # 详细：对应第一行后面的内容
        self.examples: Item = Item(name='Examples')  # 示例：对应 Examples 下的所有内容
        self.references: Item = Item(name='References')  # 参考：对应 References 下的内容

        self._obj = obj
        self._raw_doc = obj.__doc__
        self.line_number = self.get_line_number(obj)
        self.soft_link = self.get_soft_link()

        lines = self._raw_doc.split('\n')
        if lines and lines[0].startswith('@'):  # assert
            self.flag = lines[0]
            lines = lines[1:]

            # remove indent
            self._min_indent = self.get_min_indent('\n'.join(lines))
            self._lines = [ln[self._min_indent:] for ln in lines]
            self._doc = '\n'.join(self._lines)

            self.doc_parse()
            self.markdown_block = self.get_markdown_block()
            self.toc_line = self.get_toc_line(''.join(self.summary.lines))

    def get_soft_link(self):
        abs_url = inspect.getmodule(self._obj).__file__
        dirs = abs_url.split('/')
        idx = dirs[::-1].index('my')  # *从后往前*找到 my 文件夹，只有这个位置是基本固定的
        return '/'.join(dirs[-(idx + 2):])  # 再找到这个 my 文件夹的上一级目录

    @staticmethod
    def get_min_indent(s):
        """Return the minimum indentation of any non-blank line in `s`"""
        indents = [len(indent) for indent in RE_INDENT.findall(s)]
        if len(indents) > 0:
            return min(indents)
        else:
            return 0

    @staticmethod
    def _remove_white_lines(_lns):
        while len(_lns) > 0:
            if _lns[0].strip() == '':
                _lns.pop(0)
            else:
                break

        while len(_lns) > 0:
            if _lns[-1].strip() == '':
                _lns.pop()
            else:
                break

        return _lns

    def doc_parse(self):
        """"""

        def _update(end_lno=None):
            if start_lno > -1:
                if hasattr(self, name) and isinstance(getattr(self, name), Item):
                    getattr(self, name).flag = flag
                    _lines = self._remove_white_lines(lines[start_lno: end_lno])
                    getattr(self, name).lines = _lines

        # summary 特殊处理
        head = self._lines[0]
        if head.strip():
            self.summary.lines = [f'`{self._obj.__name__}`: ', head]

        lines = self._remove_white_lines(self._lines[1:])

        # details
        lno = 0
        while lno < len(lines) and lines[lno].find(':') == -1:
            lno += 1

        if lno > 0:
            tmp_lines = self._remove_white_lines(lines[:lno])
            self.details.lines = tmp_lines
            lines = self._remove_white_lines(lines[lno:])

        name = ''
        start_lno = -1
        for lno, ln in enumerate(lines):
            if not RE_SPACE.match(ln) and ln.find(':') != -1:
                _update(lno - 1)
                name, flag = ln.split(':', maxsplit=1)
                name = name.lower()
                start_lno = lno

        _update()

    @staticmethod
    def get_line_number(obj):
        """ 获取对象行号
        基于正则表达式，所以不一定保证准确

        Examples:
            # 获取失败示例
            class Test:
            >>> class Test:  # 正确答案应该是这行，但因为上面那行也能 match，所以实际返回的是上一行
            ...     ''''''
            >>> # get_line_number(Test)

        """
        return inspect.findsource(obj)[1] + 1

    def get_source_link(self):
        """"""
        return f'[source]({self.soft_link}#L{self.line_number})'

    @staticmethod
    def get_href(line):
        """"""
        return slugify(line, '-')

    @staticmethod
    def get_toc_line(line):
        """"""
        toc_line = f'[{line}](#{DocParser.get_href(line)})'
        return toc_line

    def get_markdown_block(self):
        """"""

        def _concat(lines):
            if not lines:
                return ''
            content = lines if isinstance(lines, str) else '<br>\n'.join(lines)
            return content + '\n\n'

        def _concat_code(lines):
            if not lines:
                return ''
            head = lines[0]
            min_indent = self.get_min_indent('\n'.join(lines[1:]))
            content = '\n'.join([ln[min_indent:] for ln in lines[1:]])
            return f"**{head}**\n```python\n{content}\n```" if lines else ''

        block = f'### {"".join(self.summary.lines)}\n'
        block += _concat('> ' + self.get_source_link())
        block += _concat(self.details.lines)
        block += _concat_code(self.examples.lines)
        return block


def get_line_number(obj):
    """ 获取对象行号
    基于正则表达式，所以不一定保证准确
    
    Examples:
        # 获取失败示例
        class Test:
        >>> class Test:  # 正确答案应该是这行，但因为上面那行也能 match，所以实际返回的是上一行
        ...     ''''''
        >>> # get_line_number(Test)

    """
    return inspect.findsource(obj)[1] + 1


def _fields_match(ln, fs):
    """"""
    for f in fs:
        if ln.startswith(f.name + ':'):
            return f.name

    return ''


def _test():
    """"""
    doctest.testmod()

    # ret = DocParser(simple_argparse)
    # print(ret.get_readme_block())

    # for it in ret:
    #     print(it)

    module_iter(r'/Users/huayang/workspace/my/studies/code/my/python/custom')


if __name__ == '__main__':
    """"""
    _test()
