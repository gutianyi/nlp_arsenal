# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     _tokenizer.py
   Description:
   Author:        GU Tianyi
   date:          2022/4/3
   desc:
-------------------------------------------------
"""
import numpy as np
class BaseTokenizer(object):
    """
    文本编码器基类
    """
    def __init__(self, vocab, max_seq_len):
        """
        :param vocab: 词典类对象
        :param max_seq_len: 文本最大长度
        """
        self.vocab = vocab
        self.max_seq_len = max_seq_len

    def tokenize(self, text):
        return self.vocab.tokenize(text)

    def pad_and_truncate(self, sequence, maxlen, dtype='int64', padding='post',
                         truncating='post', value=0):
        x = (np.ones(maxlen) * value).astype(dtype)
        if truncating == 'pre':
            trunc = sequence[-maxlen:]
        else:
            trunc = sequence[:maxlen]
        trunc = np.asarray(trunc, dtype=dtype)
        if padding == 'post':
            x[:len(trunc)] = trunc
        else:
            x[-len(trunc):] = trunc
        return x