#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time:
    2021-04-26 9:05 下午
    
Author:
    huayang
    
Subject:
    
"""
import os

__all__ = [
    'get_real_ext',
    'rename_to_real_ext'
]


def get_real_ext(image_path, return_is_same=False):
    """@Image Utils
    获取图像文件的真实后缀
    如果不是图片，返回后缀为 None
    该方法不能判断图片是否完整

    Args:
        image_path:
        return_is_same: 是否返回 `is_same`

    Returns:
        ext_real, is_same
        真实后缀，真实后缀与当前后缀是否相同
        如果当前文件不是图片，则 ext_real 为 None
    """
    import imghdr

    # 获取当前后缀
    ext_cur = os.path.splitext(image_path)[1]

    if ext_cur.startswith('.'):
        ext_cur = ext_cur[1:]

    # 获取真实后缀
    ext_real = imghdr.what(image_path)

    if return_is_same:
        # 是否相同
        is_same = ext_cur == ext_real or {ext_cur, ext_real} == {'jpg', 'jpeg'}

        return ext_real, is_same

    return ext_real


def rename_to_real_ext(image_path):
    """将图片重命名为真实后缀"""
    ext_real, is_same = get_real_ext(image_path, return_is_same=True)

    if is_same or ext_real is None:
        return

    prefix, _ = os.path.splitext(image_path)
    dst = '.'.join([prefix, ext_real])
    os.rename(image_path, dst)
