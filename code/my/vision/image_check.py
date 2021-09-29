#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time:
    2021-03-09 8:56 PM
    
Author:
    huayang
    
Subject:
    图片完整性检查
"""
import os
import io


def is_jpg(img_byte):
    """"""
    return img_byte[6:10] in (b'JFIF', b'Exif')


def is_png(img_byte):
    return img_byte.startswith(b'\211PNG\r\n\032\n')


def is_gif(img_byte):
    return img_byte[:6] in (b'GIF87a', b'GIF89a')


def is_bmp(img_byte):
    return img_byte.startswith(b'BM')


def is_complete_jpg(byte_str):
    b = byte_str.rstrip(b'\0\r\n')
    return b.endswith(b'\xff\xd9')


def is_complete_png(byte_str):
    b = byte_str.rstrip(b'\0\r\n')
    return b.endswith(b'\x60\x82\x00') or b.endswith(b'\x60\x82')


# def is_complete_bmp(byte_str):
#     size = int(b'0x' + byte_str[2:6][::-1], 16)
#     return size <= len(byte_str)


def is_complete_img(byte_str):
    """不一定有效，其他类型建议先转换成 jpg 或 png"""
    try:
        from PIL import Image
        Image.open(io.BytesIO(byte_str)).verify()
    except:
        return False
    return True


def is_complete(img):
    """
    判断图片是否完整

    Args:
        img: 图像路径，或二进制字符串，如 open(file, 'rb').read()、requests.get(url).content 等
    """

    if isinstance(img, str):
        with open(img, 'rb') as f:
            img = f.read()

    if is_jpg(img):
        return is_complete_jpg(img)
    elif is_png(img):
        return is_complete_png(img)
    else:
        return is_complete_img(img)


if __name__ == '__main__':
    """"""
    dir_path = '_test_data'
    for file_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file_name)

        if os.path.isdir(file_path):
            continue

        is_valid = is_complete(file_path)
        print('%s' % '\t'.join(str(it) for it in [file_name, is_valid]))
