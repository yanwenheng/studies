#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time:
   2021-07-20 12:59 上午
   
Author:
   huayang
   
Subject:
   
"""
from enum import Enum


class ExplicitEnum(Enum):
    """
    Enum with more explicit error message for missing values.
    """

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"%r is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}"
            % value
        )


def _test():
    """"""

    def _test_ExplicitEnum():
        """"""
        class TestEnum(Enum):
            A = 'a'
            B = 'b'

        try:
            x = TestEnum('c')
            print(x.value)
        except ValueError as e:
            print(e)  # 'c' is not a valid TestEnum

        class MyEnum(ExplicitEnum):  # 错误提示不一样
            A = 'a'
            B = 'b'

        try:
            x = MyEnum('c')
            print(x.value)
        except ValueError as e:
            print(e)  # 'c' is not a valid MyEnum, please select one of ['A', 'B']

    _test_ExplicitEnum()


if __name__ == '__main__':
    """"""
    _test()