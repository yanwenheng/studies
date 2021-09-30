#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time:
    2021-06-24 12:39 上午

Author: huayang

Subject: 一些自定义的 json Encoder 或 Decoder
"""

import json
import doctest

from _ctypes import PyObj_FromPtr

from my.python.serialize import obj_to_str, str_to_obj

__all__ = [
    'NoIndentEncoder',
    'AnyEncoder',
    'AnyDecoder'
]


class NoIndentEncoder(json.JSONEncoder):
    """
    对指定的对象不应用缩进

    使用方法：将不想缩进的对象用 `NoIndentEncoder.wrap` 包裹；

    注意：如果需要保存到文件，不能直接使用 json.dump，而需要使用 json.dumps + fw.write

    Examples:
        >>> o = dict(a=1, b=NoIndentEncoder.wrap([1, 2, 3]))
        >>> s = json.dumps(o, cls=NoIndentEncoder, indent=4)
        >>> print(s)  # 注意 "b" 的 列表没有展开缩进
        {
            "a": 1,
            "b": [1, 2, 3]
        }

        # >>> fw = open(r'./_out/_test_NoIndentEncoder.json', 'w', encoding='utf8')
        # >>> _ = fw.write(s)  # 写入文件
    """

    FORMAT_SPEC = '@@{}@@'

    class Value(object):
        """ Value wrapper. """

        def __init__(self, value):
            self.value = value

    def __init__(self, *args, **kwargs):
        super(NoIndentEncoder, self).__init__(*args, **kwargs)
        self.kwargs = kwargs
        del self.kwargs['indent']
        # self._replacement_map = {}  # 缓存 id(obj) -> obj
        self._no_indent_obj_ids = set()  # 使用 PyObj_FromPtr，保存 id(obj) 即可

    def default(self, o):
        if isinstance(o, NoIndentEncoder.Value):
            # self._replacement_map[id(o)] = json.dumps(o.value, **self.kwargs)
            self._no_indent_obj_ids.add(id(o))
            return self.FORMAT_SPEC.format(id(o))
        else:
            return super(NoIndentEncoder, self).default(o)

    def encode(self, o):
        result = super(NoIndentEncoder, self).encode(o)

        # for oid, tmp_str in self._replacement_map.items():
        for oid in self._no_indent_obj_ids:
            tmp_str = json.dumps(PyObj_FromPtr(oid).value, **self.kwargs)
            result = result.replace('"{}"'.format(self.FORMAT_SPEC.format(oid)), tmp_str)
        return result

    @staticmethod
    def wrap(v):
        return NoIndentEncoder.Value(v)


class AnyEncoder(json.JSONEncoder):
    """ 支持任意对象的 Encoder，如果是非 json 默认支持对象，会转为二进制字符串；
        还原时需配合 AnyDecoder 一起使用

    Examples:
        >>> from datetime import datetime
        >>> o = dict(a=1, b=datetime(2021, 1, 1, 0, 0), c=dict(d=datetime(2012, 1, 1, 0, 0)))  # datetime 不是 json 支持的对象
        >>> s = json.dumps(o, cls=AnyEncoder)
        >>> print(s)
        {"a": 1, "b": "datetime.datetime(2021, 1, 1, 0, 0)__@AnyEncoder@__gASVKgAAAAAAAACMCGRhdGV0aW1llIwIZGF0ZXRpbWWUk5RDCgflAQEAAAAAAACUhZRSlC4=", "c": {"d": "datetime.datetime(2012, 1, 1, 0, 0)__@AnyEncoder@__gASVKgAAAAAAAACMCGRhdGV0aW1llIwIZGF0ZXRpbWWUk5RDCgfcAQEAAAAAAACUhZRSlC4="}}
        >>> x = json.loads(s, cls=AnyDecoder)
        >>> x
        {'a': 1, 'b': datetime.datetime(2021, 1, 1, 0, 0), 'c': {'d': datetime.datetime(2012, 1, 1, 0, 0)}}
        >>> assert o is not x and o == x  # o 和 x 不是同一个对象，但值是相同的
    """

    FLAG = '__@AnyEncoder@__'

    def default(self, o):
        try:
            return super(AnyEncoder, self).default(o)
        except:
            return repr(o) + AnyEncoder.FLAG + obj_to_str(o)


class AnyDecoder(json.JSONDecoder):
    """"""

    @staticmethod
    def scan(o):
        """ 递归遍历 o 中的对象，如果发现 AnyEncoder 标志，则对其还原 """
        if isinstance(o, str):
            if o.find(AnyEncoder.FLAG) != -1:  # 如果字符串中存在 AnyEncoder 标识符，说明是个特殊对象
                o = str_to_obj(o.split(AnyEncoder.FLAG)[-1])  # 提取二进制字符串并转化为 python 对象
        elif isinstance(o, list):
            for i, it in enumerate(o):
                o[i] = AnyDecoder.scan(it)  # 递归调用
        elif isinstance(o, dict):
            for k, v in o.items():
                o[k] = AnyDecoder.scan(v)  # 递归调用

        return o

    def decode(self, s: str, **kwargs):
        """"""
        obj = super(AnyDecoder, self).decode(s)
        obj = AnyDecoder.scan(obj)
        return obj


def _test():
    """"""
    doctest.testmod()


if __name__ == '__main__':
    _test()
