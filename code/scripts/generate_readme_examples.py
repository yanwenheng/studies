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

from my.python.code_analysis import DocParser, module_iter, slugify
from my.python.file_utils import files_concat
from my.python.custom import simple_argparse

RE_TAG = re.compile(r'Tag: (.*?)\s')
RE_TITLE = re.compile(r'#+\s+(.*?)$')

beg_details_tmp = '<details><summary><b> {key} <a href="{url}">¶</a></b></summary>\n'
end_details = '\n</details>\n'
auto_line = '<font color="LightGrey"><i> `This README is Auto-generated` </i></font>\n'


def hn_line(line, lv=2):
    """"""
    return f'{"#" * lv} {line}'


class AlgorithmReadme:
    """"""

    def __init__(self, args):
        """"""
        self.args = args
        self.toc_head = 'Algorithm Studies'
        self.prefix = os.path.basename(args.algo_path)
        args.problems_path = os.path.join(args.algo_path, 'problems')

        problems_dt = self.parse_problems()
        append_lines = self.gen_topic_md(problems_dt)
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

    def parse_problems(self):
        """"""
        args = self.args

        problems_dt = defaultdict(list)  # {tag: file_txt_ls}
        files = os.listdir(args.problems_path)

        # 解析算法 tags
        for f in files:
            fp = os.path.join(args.problems_path, f)
            txt = open(fp, encoding='utf8').read()
            tags = RE_TAG.search(txt)
            if tags:
                tags = [tag.strip() for tag in re.split(r'[,，]]', tags.group(1))]
            else:
                tags = ['其他']

            txt = txt.rstrip() + '\n\n---'
            for tag in tags:
                problems_dt[tag].append(txt)

        # for k, v in problems_dt.items():
        #     for idx, txt in enumerate(v):
        #         if idx < len(v) - 1:
        #             v[idx] = txt.rstrip() + '\n\n---'

        return problems_dt

    def gen_topic_md(self, problems_dt):
        """生成算法专题md"""
        args = self.args

        readme_lines = [self.toc_head, '===\n', auto_line]
        append_lines = [self.toc_head, '---\n']

        for tag, problems_txts in problems_dt.items():  # noqa
            """"""
            topic_fn = f'专题-{tag}'
            index_lines = ['Index', '---']
            # readme_lines.append(f'- [{topic_fn}]({topic_fn}.md)')
            # append_lines.append(f'- [{topic_fn}]({self.prefix}/{topic_fn}.md)')
            readme_lines.append(beg_details_tmp.format(key=topic_fn, url=f'{topic_fn}.md'))
            append_lines.append(beg_details_tmp.format(key=topic_fn, url=f'{self.prefix}/{topic_fn}.md'))
            for txt in problems_txts:
                head = self.parse_head(txt)
                index_lines.append(f'- [{head}](#{slugify(head)})')
                readme_lines.append(f'- [{head}]({topic_fn}.md#{slugify(head)})')
                append_lines.append(f'- [{head}]({self.prefix}/{topic_fn}.md#{slugify(head)})')

            readme_lines.append(end_details)
            append_lines.append(end_details)
            index_lines.append('\n---')
            f_out = os.path.join(args.problems_path, '..', f'{topic_fn}.md')
            files_concat(['\n'.join(index_lines)] + problems_txts, f_out, '\n')

        with open(os.path.join(args.algo_path, 'README.md'), 'w', encoding='utf8') as fw:
            fw.write('\n'.join(readme_lines))

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

    def __init__(self, args):
        """"""
        self.args = args
        self.toc_head = 'My Code Lab'
        docs_dt = self.parse_docs()
        args.code_readme_path = os.path.join(args.code_path, 'README.md')
        self.readme_append = self.gen_readme_md(docs_dt)

    def parse_docs(self):
        """ 生成 readme for code """
        args = self.args
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

        return docs_dt

    def gen_readme_md(self, docs_dt: Dict[str, List[DocParser]]):
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
            # main_toc.append(beg_details_tmp.format(key=key, url=f'{code_prefix}/README.md#{DocParser.slugify(key)}'))

            readme_lines.append(hn_line(key, 2))
            append_lines.append(hn_line(key, 2))
            for d in blocks:
                toc.append(f'- [{d.summary_line}](#{slugify(d.summary_line)})')
                readme_lines.append(d.get_markdown_block())
                append_lines.append(d.get_markdown_block(prefix=code_prefix))

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


def gen_main_toc(toc_lines):
    """"""
    lns = ['Repo Index', '---\n']
    for ln in toc_lines:
        lns.append('- ' + DocParser.get_toc_line(ln))

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
                  "--algo_path ../../algorithm " \
                  "--module_path ../../code/my"
        sys.argv = command.split()
        _test()
