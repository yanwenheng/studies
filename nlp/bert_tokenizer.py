#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time:
    2021-01-27 11:53
    
Author:
    huayang
    
Subject:
    Bert 原生分词器，移除了兼容 python2 的内容

References:
    https://github.com/google-research/bert/blob/master/tokenization.py
"""
import os
import doctest
from collections import OrderedDict

from my.nlp.normalization import (
    is_char_cjk,
    is_char_whitespace,
    is_char_control,
    is_char_punctuation,
    remove_accents,
)

__all__ = [
    'BertTokenizer',
    'tokenizer'
]


def convert_by_vocab(vocab, items):
    """Converts a sequence of [tokens|ids] using the vocab."""
    output = []
    for item in items:
        output.append(vocab[item])
    return output


# def convert_tokens_to_ids(vocab, tokens):
#     return convert_by_vocab(vocab, tokens)
#
#
# def convert_ids_to_tokens(inv_vocab, ids):
#     return convert_by_vocab(inv_vocab, ids)


def load_vocab(vocab_file, encoding='utf8'):
    """Loads a vocabulary file into a dictionary."""
    vocab = OrderedDict()
    index = 0
    with open(vocab_file, encoding=encoding) as reader:
        while True:
            token = convert_to_unicode(reader.readline())
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab


def tokenize(text, do_lower_case=True):
    """Tokenizes a piece of text."""
    text = convert_to_unicode(text)
    text = _clean_text(text)

    # This was added on November 1st, 2018 for the multilingual and Chinese
    # models. This is also applied to the English models now, but it doesn't
    # matter since the English models were not trained on any Chinese data
    # and generally don't have any Chinese data in them (there are Chinese
    # characters in the vocabulary because Wikipedia does have some Chinese
    # words in the English Wikipedia.).
    text = _add_space_around_cjk_chars(text)

    orig_tokens = split_by_whitespace(text)
    split_tokens = []
    for token in orig_tokens:
        if do_lower_case:
            token = token.lower()
            token = remove_accents(token)
        split_tokens.extend(_split_on_punctuation(token))

    output_tokens = split_by_whitespace(" ".join(split_tokens))
    return output_tokens


def split_by_whitespace(text):
    """Runs basic whitespace cleaning and splitting on a piece of text.

    Examples:
        >>> text = '我爱python，我爱编程；I love python, I like programming.'
        >>> split_by_whitespace(text)
        ['我爱python，我爱编程；I', 'love', 'python,', 'I', 'like', 'programming.']
    """
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))


def _split_on_punctuation(text):
    """Splits punctuation on a piece of text.

    Examples:
        >>> text = '我爱python，我爱编程；I love python, I like programming.'
        >>> _split_on_punctuation(text)
        ['我爱python', '，', '我爱编程', '；', 'I love python', ',', ' I like programming', '.']
    """
    chars = list(text)
    i = 0
    start_new_word = True
    output = []
    while i < len(chars):
        char = chars[i]
        if is_char_punctuation(char):
            output.append([char])
            start_new_word = True
        else:
            if start_new_word:
                output.append([])
            start_new_word = False
            output[-1].append(char)
        i += 1

    return ["".join(x) for x in output]


def _add_space_around_cjk_chars(text):
    """
    Examples:
        >>> text = '我爱python，我爱编程；I love python, I like programming.'
        >>> _add_space_around_cjk_chars(text)
        ' 我  爱 python， 我  爱  编  程 ；I love python, I like programming.'
    """
    output = []
    for char in text:
        cp = ord(char)
        if is_char_cjk(cp):
            output.append(" ")
            output.append(char)
            output.append(" ")
        else:
            output.append(char)
    return "".join(output)


def _clean_text(text):
    """Performs invalid character removal and whitespace cleanup on text."""
    output = []
    for char in text:
        cp = ord(char)
        if cp == 0 or cp == 0xfffd or is_char_control(char):
            continue
        if is_char_whitespace(char):
            output.append(" ")
        else:
            output.append(char)
    return "".join(output)


class WordPieceTokenizer(object):
    """Runs WordPiece tokenziation."""

    def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=200):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """Tokenizes a piece of text into its word pieces.
        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.
        For example:
          input = "unaffable"
          output = ["un", "##aff", "##able"]
        Args:
          text: A single token or whitespace separated tokens. This should have
            already been passed through `BasicTokenizer.
        Returns:
          A list of wordpiece tokens.
        """

        text = convert_to_unicode(text)

        output_tokens = []
        for token in split_by_whitespace(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


class BertTokenizer(object):
    """@NLP Utils
    Bert 分词器

    Examples:
        >>> text = '我爱python，我爱编程；I love python, I like programming. Some unkword'

        # WordPiece 切分
        >>> tokens = tokenizer.tokenize(text)
        >>> assert [tokens[2], tokens[-2], tokens[-7]] == ['python', '##nk', 'program']

        # 模型输入
        >>> token_ids, segment_ids, masks = tokenizer.encode(text)
        >>> assert token_ids[:6] == [101, 2769, 4263, 9030, 8024, 2769]
        >>> assert segment_ids == [0] * len(token_ids)

        # 句对模式
        >>> txt1 = '我爱python'
        >>> txt2 = '我爱编程'
        >>> token_ids, segment_ids, masks = tokenizer.encode(txt1, txt2)
        >>> assert token_ids == [101, 2769, 4263, 9030, 102, 2769, 4263, 5356, 4923, 102]
        >>> assert segment_ids == [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

    """

    token2id_map: dict  # {token: id}
    id2token_map: dict  # {id: token}

    def __init__(self, vocab_file,
                 do_lower_case=True,
                 token_cls='[CLS]',
                 token_sep='[SEP]',
                 token_unk='[UNK]',
                 token_mask='[MASK]',
                 pad_index=0,
                 verbose=0):
        self.token2id_map = load_vocab(vocab_file)
        self.id2token_map = {v: k for k, v in self.token2id_map.items()}
        if verbose > 0:
            print(f'Vocab size={len(self.token2id_map)}')
        self.basic_tokenize = lambda text: tokenize(text, do_lower_case)
        self.word_piece_tokenize = WordPieceTokenizer(vocab=self.token2id_map).tokenize
        self._token_cls = token_cls
        self._token_sep = token_sep
        self._token_unk = token_unk
        self._pad_index = pad_index
        self._token_mask = token_mask

    def encode(self, txt1, txt2=None, max_len=None):
        tokens_txt1 = self.tokenize(txt1)
        tokens_txt2 = self.tokenize(txt2) if txt2 is not None else None
        self._truncate(tokens_txt1, tokens_txt2, max_len)
        tokens, len_txt1, len_txt2 = self._pack(tokens_txt1, tokens_txt2)

        token_id = self.convert_tokens_to_ids(tokens)
        token_type_id = [0] * len_txt1 + [1] * len_txt2
        mask = [1] * (len_txt1 + len_txt2)

        if max_len is not None:
            pad_len = max_len - len_txt1 - len_txt2
            token_id += [self._pad_index] * pad_len
            token_type_id += [0] * pad_len
            mask += [0] * pad_len

        return token_id, token_type_id, mask

    def batch_encode(self, seqs, max_len, convert_fn=None):
        """
        Args:
            seqs:
            max_len:
            convert_fn: 常用的 `np.asarray`, `torch.as_tensor`, `tf.convert_to_tensor`
        """
        tokens_ids = []
        segments_ids = []
        masks = []
        for seq in seqs:
            if isinstance(seq, str):
                tid, sid, mask = self.encode(txt1=seq, max_len=max_len)
            elif isinstance(seq, (tuple, list)):
                txt1, txt2 = seq[:2]
                tid, sid, mask = self.encode(txt1=txt1, txt2=txt2, max_len=max_len)
            else:
                raise ValueError('Assert seqs are list of txt or (txt1, txt2).')
            tokens_ids.append(tid)
            segments_ids.append(sid)
            masks.append(mask)

        if convert_fn is not None:
            tokens_ids = convert_fn(tokens_ids)
            segments_ids = convert_fn(segments_ids)
            masks = convert_fn(masks)

        return tokens_ids, segments_ids, masks

    def tokenize(self, text):
        split_tokens = []
        for token in self.basic_tokenize(text):
            for sub_token in self.word_piece_tokenize(token):
                split_tokens.append(sub_token)

        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        return convert_by_vocab(self.token2id_map, tokens)

    def convert_ids_to_tokens(self, ids):
        return convert_by_vocab(self.id2token_map, ids)

    @property
    def mask_id(self):
        return self.convert_tokens_to_ids([self._token_mask])[0]

    def _pack(self, tokens_1st, tokens_2nd=None):
        packed_tokens_1st = [self._token_cls] + tokens_1st + [self._token_sep]
        if tokens_2nd is not None:
            packed_tokens_2nd = tokens_2nd + [self._token_sep]
            return packed_tokens_1st + packed_tokens_2nd, len(packed_tokens_1st), len(packed_tokens_2nd)
        else:
            return packed_tokens_1st, len(packed_tokens_1st), 0

    @staticmethod
    def _truncate(tokens_1st, tokens_2nd, max_len):
        """"""
        if max_len is None:
            return

        if tokens_2nd is not None:
            while True:
                total_len = len(tokens_1st) + len(tokens_2nd)
                if total_len <= max_len - 3:  # 3 for [CLS] .. tokens_a .. [SEP] .. tokens_b [SEP]
                    break
                if len(tokens_1st) > len(tokens_2nd):
                    tokens_1st.pop()
                else:
                    tokens_2nd.pop()
        else:
            del tokens_1st[max_len - 2:]  # 2 for [CLS] .. tokens .. [SEP]


# 不是单例
# def get_tokenizer(vocab_file=None, **kwargs):
#     """
#
#     Args:
#         vocab_file:
#
#     Returns:
#
#     """
#     if vocab_file is None:
#         pwd = os.path.dirname(__file__)
#         vocab_file = os.path.join(pwd, '../data/vocab/vocab_21128.txt')
#
#     tokenizer = Tokenizer(vocab_file, **kwargs)
#     return tokenizer


# 模块内的变量默认为单例模式
_default_vocab_path = os.path.join(os.path.dirname(__file__), 'data/vocab_cn.txt')
tokenizer = BertTokenizer(_default_vocab_path)


def _test():
    """"""
    doctest.testmod()

    def _test_Tokenizer():  # noqa
        """"""
        text = '我爱python，我爱编程；I love python, I like programming. Some unkword'

        # 字切分（中文单字、英文单词）
        tokens = tokenize(text)
        assert [tokens[2], tokens[-2]] == ['python', 'some']

        # WordPiece 切分
        tokens = tokenizer.tokenize(text)
        assert [tokens[2], tokens[-2], tokens[-7]] == ['python', '##nk', 'program']

        # 模型输入
        token_ids, segment_ids, masks = tokenizer.encode(text)
        assert token_ids[:6] == [101, 2769, 4263, 9030, 8024, 2769]
        assert segment_ids == [0] * len(token_ids)

        # 句对模式
        txt1 = '我爱python'
        txt2 = '我爱编程'
        token_ids, segment_ids, masks = tokenizer.encode(txt1, txt2)
        assert token_ids == [101, 2769, 4263, 9030, 102, 2769, 4263, 5356, 4923, 102]
        assert segment_ids == [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

    # _test_Tokenizer()


if __name__ == '__main__':
    """"""
    _test()
