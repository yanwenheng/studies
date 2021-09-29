#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time:
    2021-08-31 6:05 下午

Author:
    huayang

Subject:

"""
import math
from typing import List, Set

from collections import OrderedDict  # _test_OrderedDict

# from sortedcontainers import SortedList

# from sortedcontainers import (
#     SortedList,
#     SortedKeyList,  # 等价于 `SortedList(key=fn)`
#     # SortedListWithKey,  # 等价于 `SortedKeyList`
#     SortedSet,  # 支持自定义排序函数 SortedSet(key=fn)
#     SortedDict,  # 基于 key 的值排序；注意不能使用关键词参数传入排序函数，如 `SortedSet(key=fn)`，因为这默认表示 {'key': fn}
#     # SortedKeysView,  # SortedDict.keys()
#     # SortedValuesView,  # SortedDict.values()
#     # SortedItemsView,  # SortedDict.items()
# )

# from sortedcollections import (
#     IndexableDict,  # 等价于 `SortedDict`
#     IndexableSet,  # 等价于 `SortedSet`
#     ValueSortedDict,  # 基于 value 的值排序
#     ItemSortedDict,  # 基于 <key, value> 的结果排序，默认必须传入 compare_fn
#     OrderedSet,  # 类似于 OrderedDict
#     SegmentList,  # 支持快速随机插入与删除
#     NearestDict,  # 支持对 key 做 nearest 召回, python 3.8+
# )


def merge_intersected_sets(src: List[Set]):
    """合并有交集的集合"""
    pool = set(map(frozenset, src))  # 去重
    groups = []
    while pool:
        groups.append(set(pool.pop()))
        while True:
            for s in pool:
                if groups[-1] & s:
                    groups[-1] |= s
                    pool.remove(s)
                    break
            else:
                break
    return groups


def list_unique(ls):
    """列表去重"""
    return list(set(ls))


def list_unique_sorted(ls):
    """列表去重，不改变顺序"""
    tmp = list_unique(ls)
    tmp.sort(key=ls.index)
    return tmp


def list_split(ls, per_size=None, n_chunk=None):
    """ [0, 1, 2, 3, 4, 5, 6] -> [[0, 1], [2, 3], [4, 5], [6]] """
    assert (per_size or n_chunk) and not (per_size and n_chunk), '`per_size` and `n_chunk` must be set only one.'

    if n_chunk is not None:
        per_size = math.ceil(len(ls) / n_chunk)

    ret = []
    for i in range(0, len(ls), per_size):
        ret.append(ls[i: i + per_size])

    return ret


def list_flatten(lss):
    """ [[0, 1], [2, 3], [4, 5], [6]] -> [0, 1, 2, 3, 4, 5, 6] """
    ret = []
    for it in lss:
        ret.extend(it)

    return ret


def _test():
    """"""
    from my.python.utils import function_test_dn
    from my.python.utils import enable_function_test
    enable_function_test()

    @function_test_dn
    def _test_OrderedDict():
        """"""
        dt = OrderedDict()
        dt['3'] = 3
        dt['2'] = 2
        dt['1'] = 1

        dt.update({'5': 5, '4': 4})
        print(dt)

    _test_OrderedDict()

    # @function_test_dn
    # def _test_SortedList():
    #     """"""
    #     sl = SortedList(key=lambda x: -x)
    #     sl.add(1)
    #     sl.add(3)
    #     sl.add(2)
    #     print(list(sl))
    #
    # _test_SortedList()


if __name__ == '__main__':
    """"""
    _test()
