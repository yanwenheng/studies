#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time: 2021-09-24 4:38 下午

Author: huayang

Subject: 自定义字典

"""
import os
import json

import doctest

from typing import *
from dataclasses import dataclass, fields
from collections import OrderedDict


# class DefaultOrderedDict(defaultdict, OrderedDict):
#
#     def __init__(self, default_factory=None, *a, **kw):
#         for cls in DefaultOrderedDict.mro()[1:-2]:
#             cls.__init__(self, *a, **kw)
#
#         super(DefaultOrderedDict, self).__init__()


class ArrayDict(OrderedDict):
    """ 数组字典（支持 slice 操作）

    Examples:
        >>> d = ArrayDict(a=1, b=2)
        >>> d
        ArrayDict([('a', 1), ('b', 2)])
        >>> d['a']
        1
        >>> d[1]
        ArrayDict([('b', 2)])
        >>> d['c'] = 3
        >>> d[0] = 100
        Traceback (most recent call last):
            ...
        TypeError: ArrayDict cannot use `int` as key.
        >>> d[1: 3]
        ArrayDict([('b', 2), ('c', 3)])
        >>> print(*d)
        a b c
        >>> d.setdefault('d', 4)
        4
        >>> print(d)
        ArrayDict([('a', 1), ('b', 2), ('c', 3), ('d', 4)])
        >>> d.pop('a')
        1
        >>> d.update({'b': 20, 'c': 30})
        >>> def f(**d): print(d)
        >>> f(**d)
        {'b': 20, 'c': 30, 'd': 4}

    """

    @property
    def tuple(self) -> Tuple[Any]:
        """
        Convert self to a tuple containing all the attributes/keys that are not ``None``.
        """
        return tuple(self.items())

    def __getitem__(self, key):
        """"""
        if isinstance(key, (int,)):
            return self.__class__.__call__([self.tuple[key]])
        elif isinstance(key, (slice,)):
            return self.__class__.__call__(list(self.tuple[key]))
        else:
            # return self[k]  # err: RecursionError
            # inner_dict = {k: v for (k, v) in self.items()}
            # return inner_dict[k]
            return super().__getitem__(key)

    def __setitem__(self, key, value):
        """"""
        if isinstance(key, (int,)):
            raise TypeError(f'{self.__class__.__name__} cannot use `{type(key).__name__}` as key.')
        else:
            super().__setitem__(key, value)


class ValueArrayDict(ArrayDict):
    """ 数组字典（支持 slice 操作），且 slice 获取的值是字典的 value

    Examples:
        >>> d = ValueArrayDict(a=1, b=2)
        >>> d
        ValueArrayDict([('a', 1), ('b', 2)])
        >>> assert d[1] == 2
        >>> d['c'] = 3
        >>> assert d[2] == 3
        >>> d[1:]
        (2, 3)
        >>> print(*d)  # 注意打印的是 values
        1 2 3
        >>> del d['a']
        >>> d.update({'a':10, 'b': 20})
        >>> d
        ValueArrayDict([('b', 20), ('c', 3), ('a', 10)])

    """

    @property
    def tuple(self) -> Tuple[Any]:
        """
        Convert self to a tuple containing all the attributes/keys that are not ``None``.
        """
        return tuple(self.values())

    def __getitem__(self, key):
        """"""
        if isinstance(key, (int, slice)):
            return self.tuple[key]
        else:
            # return self[k]  # err: RecursionError
            # inner_dict = {k: v for (k, v) in self.items()}
            # return inner_dict[k]
            return super().__getitem__(key)

    # def setdefault(self, *args, **kwargs):
    #     """ 不支持 setdefault 操作 """
    #     raise Exception(f"Cannot use ``setdefault`` on a {self.__class__.__name__} instance.")

    # def pop(self, *args, **kwargs):
    #     """ 不支持 pop 操作 """
    #     raise Exception(f"Cannot use ``pop`` on a {self.__class__.__name__} instance.")

    # def update(self, *args, **kwargs):
    #     """ 不支持 update 操作 """
    #     raise Exception(f"Cannot use ``update`` on a {self.__class__.__name__} instance.")

    def __iter__(self):
        """ dict 默认迭代的对象是 keys，重写使迭代 values

        Examples:
            >>> sd = ValueArrayDict(a=1, b=2)
            >>> # 没有重写 __iter__ 时：
            >>> # print(*sd)  # a b
            >>> # 重写 __iter__ 后：
            >>> print(*sd)
            1 2

        """
        return iter(self.tuple)


class BunchDict(dict):
    """
    基于 dict 实现 Bunch 模式
    行为上类似于 argparse.Namespace，但可以使用 dict 的方法，更通用

    Examples:

        >>> c = BunchDict(a=1, b=2)
        >>> c
        {'a': 1, 'b': 2}
        >>> c.c = 3
        >>> c
        {'a': 1, 'b': 2, 'c': 3}
        >>> dir(c)
        ['a', 'b', 'c']
        >>> assert 'a' in c
        >>> del c.a
        >>> assert 'a' not in c

        >>> x = BunchDict(d=4, e=c)
        >>> x
        {'d': 4, 'e': {'b': 2, 'c': 3}}
        >>> z = {'d': 4, 'e': {'a': 1, 'b': 2, 'c': 3}}
        >>> y = BunchDict.from_dict(z)
        >>> y
        {'d': 4, 'e': {'a': 1, 'b': 2, 'c': 3}}

    References:
        - bunch（pip install bunch）
    """

    # 最简单实现 Bunch 模式的方法，可以不用重写 __setattr__ 等方法
    # def __init__(self, *args, **kwargs):
    #     super(BunchDict, self).__init__(*args, **kwargs)
    #     self.__dict__ = self

    def __dir__(self):
        """ 屏蔽其他属性或方法 """
        return self.keys()

    def __getattr__(self, key):
        """ 使 o.key 等价于 o[key] """
        try:
            return object.__getattribute__(self, key)
        except AttributeError:
            try:
                return self[key]
            except KeyError:
                raise AttributeError(key)

    def __setattr__(self, name, value):
        """ 使 o.name = value 等价于 o[name] = value """
        try:
            # Throws exception if not in prototype chain
            object.__getattribute__(self, name)
        except AttributeError:
            self[name] = value
        else:
            object.__setattr__(self, name, value)

    def __delattr__(self, key):
        """ 支持 del x.y """
        try:
            # Throws exception if not in prototype chain
            object.__getattribute__(self, key)
        except AttributeError:
            try:
                del self[key]
            except KeyError:
                raise AttributeError(key)
        else:
            object.__delattr__(self, key)

    # 继承自 dict，所以不需要本方法
    # def to_dict(self):
    #     return _unbunch(self)

    @classmethod
    def from_dict(cls, d: dict):
        return _bunch(d, cls)


class BunchArrayDict(ArrayDict, BunchDict):
    """ 
    
    Examples:
        >>> d = BunchArrayDict(a=1, b=2)
        >>> isinstance(d, dict)
        True
        >>> print(d, d.a, d[1])
        BunchArrayDict([('a', 1), ('b', 2)]) 1 BunchArrayDict([('b', 2)])
        >>> d.a, d.b, d.c = 10, 20, 30
        >>> print(d, d[1:])
        BunchArrayDict([('a', 10), ('b', 20), ('c', 30)]) BunchArrayDict([('b', 20), ('c', 30)])
        >>> print(*d)
        a b c
        >>> dir(d)
        ['a', 'b', 'c']
        >>> assert 'a' in d
        >>> del d.a
        >>> assert 'a' not in d
        >>> getattr(d, 'a', 100)
        100

        # 测试嵌套
        >>> x = BunchArrayDict(d=40, e=d)
        >>> x
        BunchArrayDict([('d', 40), ('e', BunchArrayDict([('b', 20), ('c', 30)]))])
        >>> print(x.d, x.e.b)
        40 20

        >>> z = {'d': 4, 'e': {'a': 1, 'b': 2, 'c': 3}}
        >>> y = BunchArrayDict.from_dict(z)
        >>> y
        BunchArrayDict([('d', 4), ('e', BunchArrayDict([('a', 1), ('b', 2), ('c', 3)]))])
        >>> y.e.c
        3

    """


class BunchValueArrayDict(ValueArrayDict, BunchDict):
    """

    Examples:
        >>> d = BunchValueArrayDict(a=1, b=2)
        >>> isinstance(d, dict)
        True
        >>> print(d, d.a, d[1])
        BunchValueArrayDict([('a', 1), ('b', 2)]) 1 2
        >>> d.a, d.b, d.c = 10, 20, 30
        >>> print(d, d[2], d[1:])
        BunchValueArrayDict([('a', 10), ('b', 20), ('c', 30)]) 30 (20, 30)
        >>> print(*d)
        10 20 30
        >>> dir(d)
        ['a', 'b', 'c']
        >>> assert 'a' in d
        >>> del d.a
        >>> assert 'a' not in d
        >>> getattr(d, 'a', 100)
        100

        # 测试嵌套
        >>> x = BunchValueArrayDict(d=40, e=d)
        >>> x
        BunchValueArrayDict([('d', 40), ('e', BunchValueArrayDict([('b', 20), ('c', 30)]))])
        >>> print(x.d, x.e.b)
        40 20

        >>> z = {'d': 4, 'e': {'a': 1, 'b': 2, 'c': 3}}
        >>> y = BunchValueArrayDict.from_dict(z)
        >>> y
        BunchValueArrayDict([('d', 4), ('e', BunchValueArrayDict([('a', 1), ('b', 2), ('c', 3)]))])
        >>> y.e.c
        3

    """


@dataclass()
class Fields(BunchDict):
    """

    Examples:
        >>> @dataclass()
        ... class Test(Fields):
        ...     c1: str = 'c1'
        ...     c2: int = 0
        ...     c3: list = None

        >>> r = Test()
        >>> r
        Test(c1='c1', c2=0, c3=None)
        >>> r.c1  # r[0]
        'c1'

        >>> r = Test(c1='a', c3=[1,2,3])
        >>> r.c1
        'a'
        >>> r.c3
        [1, 2, 3]

        >>> d = {'c1': 'C1', 'c2': 10, 'c3': [1, 2]}
        >>> t = Test(**d)
        >>> t.c4 = 1  # 不推荐新增 attr
        >>> t.c4
        1
        >>> t  # 没有 c4
        Test(c1='C1', c2=10, c3=[1, 2])
        >>> list(t.items())  # 这里有 c4
        [('c1', 'C1'), ('c2', 10), ('c3', [1, 2]), ('c4', 1)]

    """

    def __post_init__(self):
        """"""
        # 获取所有 field
        class_fields = fields(self)
        # 依次添加到 dict 中
        for f in class_fields:
            self[f.name] = getattr(self, f.name)


@dataclass()
class ArrayFields(Fields, BunchValueArrayDict):
    """
    References:
        transformers.file_utils.ModelOutput

    Examples:
        >>> @dataclass()
        ... class Test(ArrayFields):
        ...     c1: str = 'c1'
        ...     c2: int = 0
        ...     c3: list = None

        >>> r = Test()
        >>> r
        Test(c1='c1', c2=0, c3=None)
        >>> r.tuple
        ('c1', 0, None)
        >>> r.c1  # r[0]
        'c1'
        >>> r[1]  # r.c2
        0
        >>> r[1:]
        (0, None)

        >>> r = Test(c1='a', c3=[1,2,3])
        >>> r.c1
        'a'
        >>> r[-1]
        [1, 2, 3]

    """


class Config(BunchDict):
    """
    配置类基类，继承使用，支持保存为 json 文件，且支持保存默认 json 不支持的对象类型

    Examples:
        # _TestConfig 继承自 BaseConfig，并对配置项设置默认值
        >>> class _TestConfig(Config):
        ...     def __init__(self, **config_items):
        ...         from datetime import datetime
        ...         self.a = 1
        ...         self.b = datetime(2012, 1, 1)  # 注意是一个特殊对象，默认 json 是不支持的
        ...         super(_TestConfig, self).__init__(**config_items)
        
        >>> args = _TestConfig()
        >>> assert args.a == 1  # 默认值
        >>> args.a = 10  # 修改值
        >>> assert args.a == 10  # 自定义值

        >>> args = _TestConfig(a=10)  # 创建时修改
        >>> assert args.a == 10

        # 添加默认中不存的配置项
        >>> args.c = 3  # 默认中没有的配置项（不推荐，建议都定义在继承类中，并设置默认值）
        >>> assert args.c == 3
        >>> print(args)  # 注意 'b' 保存成了特殊形式
        _TestConfig: {
            "a": 10,
            "b": "datetime.datetime(2012, 1, 1, 0, 0)__@AnyEncoder@__gASVKgAAAAAAAACMCGRhdGV0aW1llIwIZGF0ZXRpbWWUk5RDCgfcAQEAAAAAAACUhZRSlC4=",
            "c": 3
        }

        # 保存配置到文件
        >>> fp = r'./-test/test_save_config.json'
        >>> os.makedirs(os.path.dirname(fp), exist_ok=True)
        >>> args.save(fp)  # 保存
        >>> x = _TestConfig.load(fp)  # 重新加载
        >>> assert x.dict == args.dict
        >>> _ = os.system('rm -rf ./-test')

    """

    def __str__(self):
        """"""
        return f'{self.__class__.__name__}: {self.print_dict}'

    @property
    def dict(self):
        """"""
        return self

    @property
    def print_dict(self):
        """"""
        from my.python.custom import AnyEncoder
        return json.dumps(self.dict, cls=AnyEncoder, indent=4, ensure_ascii=False, sort_keys=True)

    def save(self, fp: str):
        """ 保存配置到文件 """
        with open(fp, 'w', encoding='utf8') as fw:
            fw.write(self.print_dict)

    @classmethod
    def load(cls, fp: str):
        """"""
        from my.python.custom import AnyDecoder
        config_items = json.load(open(fp, encoding='utf8'), cls=AnyDecoder)
        return cls(**config_items)


def _bunch(x, cls):
    """ Recursively transforms a dictionary into a Bunch via copy.

        >>> b = _bunch({'urmom': {'sez': {'what': 'what'}}}, BunchDict)
        >>> b.urmom.sez.what
        'what'

        bunchify can handle intermediary dicts, lists and tuples (as well as
        their subclasses), but ymmv on custom datatypes.

        >>> b = _bunch({ 'lol': ('cats', {'hah':'i win'}), 'hello': [{'french':'salut', 'german':'hallo'}]}, BunchDict)
        >>> b.hello[0].french
        'salut'
        >>> b.lol[1].hah
        'i win'

        nb. As dicts are not hashable, they cannot be nested in sets/frozensets.
    """
    if isinstance(x, dict):
        return cls((k, _bunch(v, cls)) for k, v in x.items())
    elif isinstance(x, (list, tuple)):
        return type(x)(_bunch(v, cls) for v in x)
    else:
        return x


def _unbunch(x):  # noqa
    """ Recursively converts a Bunch into a dictionary.

        >>> b = BunchDict(foo=BunchDict(lol=True), hello=42, ponies='are pretty!')
        >>> _unbunch(b)
        {'foo': {'lol': True}, 'hello': 42, 'ponies': 'are pretty!'}

        unbunchify will handle intermediary dicts, lists and tuples (as well as
        their subclasses), but ymmv on custom datatypes.

        >>> b = BunchDict(foo=['bar', BunchDict(lol=True)], hello=42, ponies=('pretty!', BunchDict(lies='trouble!')))
        >>> _unbunch(b)
        {'foo': ['bar', {'lol': True}], 'hello': 42, 'ponies': ('pretty!', {'lies': 'trouble!'})}

        nb. As dicts are not hashable, they cannot be nested in sets/frozensets.
    """
    if isinstance(x, dict):
        return dict((k, _unbunch(v)) for k, v in x.items())
    elif isinstance(x, (list, tuple)):
        return type(x)(_unbunch(v) for v in x)
    else:
        return x


def _test():
    """"""
    doctest.testmod()


if __name__ == '__main__':
    """"""
    _test()
