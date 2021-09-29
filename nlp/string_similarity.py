#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time: 2021-08-17 4:26 下午

Author: huayang

Subject:
    浅层相似度，基于纯文本度量
"""
import doctest
import warnings
from math import floor

try:
    import Levenshtein  # noqa
    from gensim.similarities.levenshtein import levdist
    from gensim.similarities.levenshtein import levsim
except ImportError:
    msg = (
        "Levenshtein package <https://pypi.org/project/python-Levenshtein/> is unavailable. "
        "Install Levenhstein (e.g. `pip install python-Levenshtein`) to suppress this warning."
    )
    warnings.warn(msg)

INF = float("inf")


def bm25_similarity(t1, t2):
    """"""


def jaccard_similarity(a, b):  # noqa
    """
    Examples:
        >>> jaccard_similarity('abc', 'cba')
        1.0
        >>> jaccard_similarity('abc', 'bcd')
        0.5

    Args:
        a:
        b:
    """
    set_a = set(a)
    set_b = set(b)

    if not (set_a and set_b):
        return 0.0
    return 1.0 * len(set_a & set_b) / len(set_a | set_b)


def levenshtein_distance(t1, t2, normalized=False):
    """Compute absolute Levenshtein distance of two strings.

    Parameters
    ----------
    t1 : {bytes, str, unicode}
        The first compared term.
    t2 : {bytes, str, unicode}
        The second compared term.
    normalized :

    Returns
    -------
    float
        The Levenshtein distance between `t1` and `t2`.
    """
    if t1 == t2:
        return 0.0

    if normalized:
        max_len = max(len(t1), len(t2))
    else:
        max_len = 1

    return 1.0 * Levenshtein.distance(t1, t2) / max_len


def levenshtein_similarity(t1, t2):
    """Get the Levenshtein similarity between two terms.

    Return the Levenshtein similarity between two terms. The similarity is a
    number between <0.0, 1.0>, higher is more similar.

    Examples:
        >>> round(levenshtein_similarity('abc', 'abd'), 3)
        0.667
        >>> round(levenshtein_similarity('abd', 'abc'), 3)
        0.667
        >>> round(levenshtein_similarity('abc', 'abc'), 3)
        1.0
        >>> round(levenshtein_similarity('abc', ''), 3)
        0.0
        >>> round(levenshtein_similarity('', ''), 3)
        1.0

    Parameters
    ----------
    t1 : {bytes, str, unicode}
        The first compared term.
    t2 : {bytes, str, unicode}
        The second compared term.

    Returns
    -------
    float
        The Levenshtein similarity between `t1` and `t2`.
    """
    return 1.0 - levenshtein_distance(t1, t2, normalized=True)


def _test():
    """"""
    doctest.testmod()


if __name__ == '__main__':
    """"""
    _test()
