#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time: 2021-10-01 11:13 下午

Author: huayang

Subject:

"""
import os
import re
import sys
import json
import inspect
import doctest

from types import *
from typing import *
from collections import defaultdict, OrderedDict
from pkgutil import walk_packages
from dataclasses import dataclass

from sortedcontainers import SortedList

try:
    script_path = os.path.dirname(__file__)
    code_path = os.path.join(script_path, '..')
    repo_path = os.path.join(code_path, '..')
    # print(os.path.basename(os.path.abspath(repo_path)))
    sys.path.append(code_path)
except:  # noqa
    pass

from my.python.code_analysis import module_iter, slugify
from my.python.file_utils import files_concat
from my.python.custom import simple_argparse

RE_TAG = re.compile(r'Tag: (.*?)\s')
RE_TITLE = re.compile(r'#+\s+(.*?)$')
RE_INDENT = re.compile(r'^([ ]*)(?=\S)', re.MULTILINE)

beg_details_tmp = '<details><summary><b> {key} <a href="{url}">¶</a></b></summary>\n'
beg_details_cnt_tmp = '<details><summary><b> {key} ({cnt}) <a href="{url}">¶</a></b></summary>\n'
end_details = '\n</details>\n'
auto_line = '<font color="LightGrey"><i> `This README is Auto-generated` </i></font>\n'

tag_map = [  # 文件名: tag名
    # ('滑动窗口(双指针)', '滑动窗口'),
    ('双指针(滑动窗口)', '双指针(滑动窗口)'),
    ('双指针(滑动窗口)', '滑动窗口'),
    ('双指针(滑动窗口)', '双指针(首尾)'),
    ('双指针(滑动窗口)', '首尾双指针'),
    ('深度优先搜索(递归)', '深度优先搜索(递归)'),
    ('深度优先搜索(递归)', '深度优先搜索'),
    ('深度优先搜索(递归)', 'dfs'),
    ('递归(迭代)', '递归(迭代)'),
    ('递归(迭代)', '递归'),
    ('递归(迭代)', '递归/迭代'),
    ('递归(迭代)', '递归-迭代'),
    ('哈希表', '哈希表'),
    ('哈希表', 'hash'),
    ('链表', '链表'),
    ('二叉树(树)', '二叉树(树)'),
    ('二叉树(树)', '二叉树'),
    ('二叉树(树)', '树'),
    ('前缀和', '前缀和'),
    ('字符串', '字符串'),
    ('位运算', '位运算'),
    ('二分查找', '二分查找'),
    ('二分查找', '二分搜索'),
    ('模拟', '模拟'),
    ('数学', '数学'),
    ('其他', '其他'),
    ('其他', 'other'),
]

tag_dt = {k: v for v, k in tag_map}


def hn_line(line, lv=2):
    """"""
    return f'{"#" * lv} {line}'


class AlgorithmReadme:
    """"""

    def __init__(self, args):
        """"""
        self.args = args
        self.toc_head = 'Algorithm Studies'
        self.prefix_topics = 'topics'  # os.path.basename(args.algo_path)
        self.prefix_algo = os.path.basename(os.path.abspath(args.algo_path))
        self.prefix_repo = os.path.join(os.path.basename(args.algo_path), self.prefix_topics)
        args.problems_path = os.path.join(args.algo_path, 'problems')

        problems_dt = self.parse_problems()
        append_lines = self.gen_topic_md_sorted(problems_dt)
        self.readme_append = '\n'.join(append_lines)

        # algo_path = os.path.join(repo_path, self.prefix)
        # fns = sorted([fn for fn in os.listdir(algo_path) if fn.startswith('专题-')])

        # toc_lns = [self.toc_head, '---']
        # for fn in fns:
        #     name, _ = os.path.splitext(fn)
        #     ln = f'- [{name}]({os.path.join(self.prefix, fn)})'
        #     toc_lns.append(ln)
        #
        # self.toc = '\n'.join(toc_lns)

    def gen_tags_svg(self, tags):  # noqa
        """"""
        lns = ['\n']
        for idx, (tag, topic) in enumerate(tags.items()):
            """"""
            # ![ForgiveDB](https://img.shields.io/badge/ForgiveDB-HuiZ-brightgreen.svg)
            lns.append(f'[![{tag}](https://img.shields.io/badge/{tag}-lightgray.svg)]({self.get_topic_fn(topic)})')
            # lns.append(f'[{tag}](https://img.shields.io/badge/{tag}-lightgray.svg)]')

        return '\n'.join(lns)

    def parse_problems(self):
        """"""
        args = self.args

        problems_dt = defaultdict(list)  # {tag: file_txt_ls}
        # files = os.listdir(args.problems_path)

        file_iter = []
        for prefix, _, files in os.walk(args.problems_path):
            for f in files:
                fn, ext = os.path.splitext(f)
                if ext != '.md':
                    continue

                fp = os.path.join(prefix, f)
                suffix = '-'.join(prefix.split('/')[-2:])
                file_iter.append((fn, fp, suffix))

        # 解析算法 tags
        for fn, fp, suffix in file_iter:
            # fn, _ = os.path.splitext(f)
            # fp = os.path.join(args.problems_path, f)
            txt = open(fp, encoding='utf8').read()
            tags = RE_TAG.search(txt)
            if tags:
                tags = re.split(r'[,，、]', tags.group(1))
                tag2topic = {tag.strip(): tag_dt[tag.strip().lower()] for tag in tags}
                topics = list(tag2topic.values())
            else:
                tag2topic = {'其他': '其他'}
                topics = ['其他']

            src, lv, pid, pn = fn.split('_')
            head = f'{pn} ({src}, {lv}, No.{pid}, {suffix})'
            lines = txt.split('\n')
            lines[0] = f'### {head}'
            lines.insert(1, self.gen_tags_svg(tag2topic))
            txt = '\n'.join(lines)
            txt = txt.rstrip().replace(r'../../../_assets', '../_assets') + '\n\n---'
            for topic in topics:
                problems_dt[topic].append((head, txt))

        for k, v in problems_dt.items():
            problems_dt[k] = sorted(v)

        problems_dt = OrderedDict(sorted(problems_dt.items()))
        return problems_dt

    @staticmethod
    def get_topic_fn(tag):
        return f'专题-{tag}.md'

    def gen_topic_md(self, problems_dt):
        """生成算法专题md"""
        args = self.args

        readme_lines = [self.toc_head, '===\n', auto_line]
        append_lines = [self.toc_head, '---\n']

        for tag, problems_txts in problems_dt.items():  # noqa
            """"""
            topic_fn = self.get_topic_fn(tag)
            topic_name, _ = os.path.splitext(topic_fn)
            index_lines = ['Index', '---']
            # readme_lines.append(f'- [{topic_fn}]({topic_fn}.md)')
            # append_lines.append(f'- [{topic_fn}]({self.prefix}/{topic_fn}.md)')
            algo_url = os.path.join(self.prefix_topics, topic_fn)
            repo_url = os.path.join(self.prefix_repo, topic_fn)
            readme_lines.append(beg_details_tmp.format(key=topic_name, url=algo_url))
            append_lines.append(beg_details_tmp.format(key=topic_name, url=repo_url))

            contents = []
            for (head, txt) in problems_txts:
                # head = fn
                # link = self.parse_head(txt)
                link = slugify(head)
                contents.append(txt)
                index_lines.append(f'- [{head}](#{link})')
                readme_lines.append(f'- [{head}]({algo_url}#{link})')
                append_lines.append(f'- [{head}]({repo_url}#{link})')

            readme_lines.append(end_details)
            append_lines.append(end_details)
            index_lines.append('\n---')
            f_out = os.path.join(args.repo_path, self.prefix_repo, topic_fn)
            files_concat(['\n'.join(index_lines)] + contents, f_out, '\n')

        with open(os.path.join(args.algo_path, 'README.md'), 'w', encoding='utf8') as fw:
            fw.write('\n'.join(readme_lines))

        return append_lines

    def gen_topic_md_sorted(self, problems_dt, top=10):
        """生成算法专题md，对主页topics排序"""
        args = self.args

        readme_lines = [self.toc_head, '===\n', auto_line]
        append_lines = [self.toc_head, '---\n']

        append_blocks = []

        for tag, problems_txts in problems_dt.items():  # noqa
            """"""
            append_tmp = []
            topic_fn = self.get_topic_fn(tag)
            topic_name, _ = os.path.splitext(topic_fn)
            index_lines = ['Index', '---']
            # readme_lines.append(f'- [{topic_fn}]({topic_fn}.md)')
            # append_lines.append(f'- [{topic_fn}]({self.prefix}/{topic_fn}.md)')
            algo_url = os.path.join(self.prefix_topics, topic_fn)
            repo_url = os.path.join(self.prefix_repo, topic_fn)
            problems_cnt = len(problems_txts)

            readme_lines.append(beg_details_cnt_tmp.format(key=topic_name, url=algo_url, cnt=problems_cnt))
            # append_lines.append(beg_details_tmp.format(key=topic_name, url=repo_url))
            append_tmp.append(beg_details_cnt_tmp.format(key=topic_name, url=repo_url, cnt=problems_cnt))

            contents = []
            for (head, txt) in problems_txts:
                # head = fn
                # link = self.parse_head(txt)
                link = slugify(head)
                contents.append(txt)
                index_lines.append(f'- [{head}](#{link})')
                readme_lines.append(f'- [{head}]({algo_url}#{link})')
                # append_lines.append(f'- [{head}]({repo_url}#{link})')
                append_tmp.append(f'- [{head}]({repo_url}#{link})')

            readme_lines.append(end_details)
            # append_lines.append(end_details)
            append_tmp.append(end_details)
            index_lines.append('\n---')
            f_out = os.path.join(args.repo_path, self.prefix_repo, topic_fn)
            files_concat(['\n'.join(index_lines)] + contents, f_out, '\n')

            append_blocks.append((problems_cnt, append_tmp))

        with open(os.path.join(args.algo_path, 'README.md'), 'w', encoding='utf8') as fw:
            fw.write('\n'.join(readme_lines))

        append_blocks = sorted(append_blocks, key=lambda x: -x[0])
        for _, block in append_blocks[:top]:
            append_lines += block

        append_lines.append('<details><summary><b> Others ... <a href="{url}">¶</a></b></summary>\n'.format(
            url=f'{self.prefix_algo}/README.md'
        ))

        for _, block in append_blocks[top:]:
            append_lines += block

        append_lines.append(end_details)

        # append_lines.append(f'- [All Topics]({self.prefix_algo}/README.md)')
        return append_lines

    @staticmethod
    def parse_head(txt):
        """"""
        # 标题解析
        try:
            head = RE_TITLE.search(txt.split('\n', maxsplit=1)[0]).group(1)
        except:
            raise Exception('parsing head error!')

        return head


class CodeReadme:
    """"""

    @dataclass()
    class DocItem:
        """ 每个 docstring 需要提取的内容 """
        flag: str
        summary: str
        content: str
        module_path: str
        line_no: int
        link: str = None

        def __post_init__(self):
            self.link = f'[source]({self.module_path}#L{self.line_no})'

        def get_block(self, prefix=''):
            """"""

            block = f'### {self.summary}\n'
            block += f'> [source]({os.path.join(prefix, self.module_path)}#L{self.line_no})\n\n'
            block += f'<details><summary><b> Intro & Example </b></summary>\n\n'
            block += '```python\n'
            block += f'{self.content}'
            block += '```\n\n'
            block += '</details>\n'

            return block

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

    def __init__(self, args):
        """"""
        self.args = args
        self.toc_head = 'My Code Lab'
        docs_dt = self.parse_docs()
        self.code_basename = os.path.basename(os.path.abspath(args.code_path))
        args.code_readme_path = os.path.join(args.code_path, 'README.md')
        self.readme_append = self.gen_readme_md_simply(docs_dt)

    def parse_docs(self):
        """ 生成 readme for code """
        args = self.args
        docs_dt = defaultdict(list)

        for module in module_iter(args.code_path):
            if hasattr(module, '__all__'):
                for obj_str in module.__all__:
                    obj = getattr(module, obj_str)
                    if isinstance(obj, (ModuleType, FunctionType, type)) \
                            and getattr(obj, '__doc__') \
                            and obj.__doc__.startswith('@'):
                        doc = self.parse_doc(obj)
                        docs_dt[doc.flag].append(doc)

        return docs_dt

    def parse_doc(self, obj) -> DocItem:
        """"""
        raw_doc = obj.__doc__
        lines = raw_doc.split('\n')
        flag = lines[0][1:]

        lines = lines[1:]
        min_indent = self.get_min_indent('\n'.join(lines))
        lines = [ln[min_indent:] for ln in lines]

        summary = f'`{obj.__name__}`: {lines[0]}'
        content = '\n'.join(lines)

        line_no = self.get_line_number(obj)
        module_path = self.get_module_path(obj)
        return self.DocItem(flag, summary, content, module_path, line_no)

    @staticmethod
    def get_module_path(obj):
        abs_url = inspect.getmodule(obj).__file__
        dirs = abs_url.split('/')
        idx = dirs[::-1].index('my')  # *从后往前*找到 my 文件夹，只有这个位置是基本固定的
        return '/'.join(dirs[-(idx + 1):])  # 再找到这个 my 文件夹的上一级目录

    @staticmethod
    def get_min_indent(s):
        """Return the minimum indentation of any non-blank line in `s`"""
        indents = [len(indent) for indent in RE_INDENT.findall(s)]
        if len(indents) > 0:
            return min(indents)
        else:
            return 0

    def gen_readme_md(self, docs_dt: Dict[str, List[DocItem]]):
        """"""
        args = self.args
        code_prefix = os.path.basename(os.path.abspath(args.code_path))
        # print(code_prefix)

        toc = [self.toc_head, '---\n']
        # main_toc = ['My Code Lab', '---\n']
        readme_lines = []
        append_lines = []

        key_sorted = sorted(docs_dt.keys())
        for key in key_sorted:
            blocks = docs_dt[key]
            toc.append(beg_details_tmp.format(key=key, url=f'#{slugify(key)}'))

            readme_lines.append(hn_line(key, 2))
            append_lines.append(hn_line(key, 2))
            for d in blocks:
                toc.append(f'- [{d.summary}](#{slugify(d.summary)})')
                readme_lines.append(d.get_block())
                append_lines.append(d.get_block(prefix=code_prefix))

            toc.append(end_details)
            # main_toc.append(end_details)

        toc_str = '\n'.join(toc[:2] + [auto_line] + toc[2:])
        sep = '\n---\n\n'
        content_str = '\n\n'.join(readme_lines)
        code_readme = toc_str + sep + content_str
        with open(args.code_readme_path, 'w', encoding='utf8') as fw:
            fw.write(code_readme)

        toc_str = '\n'.join(toc)
        main_append = toc_str + sep + '\n\n'.join(append_lines)
        return main_append

    def gen_readme_md_simply(self, docs_dt: Dict[str, List[DocItem]]):
        """ 简化首页的输出 """
        args = self.args
        # code_prefix = os.path.basename(os.path.abspath(args.code_path))
        # print(code_prefix)

        toc = [self.toc_head, '---\n']
        append_toc = [self.toc_head, '---\n']
        # main_toc = ['My Code Lab', '---\n']
        readme_lines = []
        # append_lines = []

        key_sorted = sorted(docs_dt.keys())
        for key in key_sorted:
            blocks = docs_dt[key]
            toc.append(beg_details_tmp.format(key=key, url=f'#{slugify(key)}'))
            append_toc.append(beg_details_tmp.format(key=key, url=f'{self.code_basename}/README.md#{slugify(key)}'))

            readme_lines.append(hn_line(key, 2))
            # append_lines.append(hn_line(key, 2))
            for d in blocks:
                toc.append(f'- [{d.summary}](#{slugify(d.summary)})')
                append_toc.append(f'- [{d.summary}]({self.code_basename}/README.md#{slugify(d.summary)})')
                readme_lines.append(d.get_block())
                # append_lines.append(d.get_block(prefix=code_prefix))

            toc.append(end_details)
            # main_toc.append(end_details)
            append_toc.append(end_details)

        toc_str = '\n'.join(toc[:2] + [auto_line] + toc[2:])
        sep = '\n---\n\n'
        content_str = '\n\n'.join(readme_lines)
        code_readme = toc_str + sep + content_str
        with open(args.code_readme_path, 'w', encoding='utf8') as fw:
            fw.write(code_readme)

        append_toc_str = '\n'.join(append_toc)
        main_append = append_toc_str + sep  # + '\n\n'.join(append_lines)
        return main_append


def gen_main_toc(toc_lines):
    """"""

    def get_toc_line(line, prefix=''):
        """"""
        toc_line = f'[{line}]({prefix}#{slugify(line)})'
        return toc_line

    lns = ['Repo Index', '---\n']
    for ln in toc_lines:
        lns.append('- ' + get_toc_line(ln))

    return lns


def pipeline():
    """"""
    args = simple_argparse()
    args.repo_readme_path = os.path.join(args.repo_path, r'README.md')
    if os.path.exists(args.repo_readme_path):
        readme_old = open(args.repo_readme_path, encoding='utf8').read()
    else:
        readme_old = ''
    # code_toc, code_append = gen_code_readme(args)

    cr = CodeReadme(args)
    ar = AlgorithmReadme(args)

    block_toc = gen_main_toc([ar.toc_head, cr.toc_head])
    readme_main_path = os.path.join(repo_path, r'README-main.md')

    main_auto_line = '<font color="LightGrey"><i> `The following is Auto-generated` </i></font>'
    files_concat(src_in=[readme_main_path,
                         main_auto_line,
                         '\n'.join(block_toc),
                         ar.readme_append,
                         cr.readme_append],
                 file_out=args.repo_readme_path,
                 sep='\n---\n\n')
    readme = open(args.repo_readme_path, encoding='utf8').read()
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
        command = "generate_readme_examples.py " \
                  "--repo_path ../../ " \
                  "--code_path ../../code/ " \
                  "--algo_path ../../algorithm "
        sys.argv = command.split()
        _test()
