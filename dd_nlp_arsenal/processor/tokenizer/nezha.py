# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     transformer.py
   Description:
   Author:        GU Tianyi
   date:          2022/4/3
   desc:
-------------------------------------------------
"""
import transformers
from dd_nlp_arsenal.processor.tokenizer._tokenizer import BaseTokenizer


class TransformerTokenizer(BaseTokenizer):
    def __init__(self, vocab, max_seq_len):
        """
        transformers文本编码器，用于对文本进行分词、ID化、填充等操作
        :param vocab:   transformers词典类对象、词典地址或词典名，用于实现文本分词和ID化
        :param max_seq_len: int 预设的文本最大长度
        """
        super(TransformerTokenizer, self).__init__(vocab, max_seq_len)

        if isinstance(vocab, str):
            # TODO: 改成由自定义的字典所决定
            vocab = transformers.BertTokenizer.from_pretrained(vocab)
        else:
            ValueError('vocab should be a path string')

        self.vocab = vocab
        self.additional_special_tokens = set()
        self.tokenizer_type = 'transformer'


    def sequence_to_ids(self, sequence):
        return self.vocab.encode_plus(
            sequence,
            add_special_tokens=True,
            max_length=self.max_seq_len,
            return_token_type_ids=True,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )



class SentenceTokenizer(TransformerTokenizer):
    """
    Transfomer文本编码器，用于单句子进行分词、ID化、填充等操作
    Args:
        vocab: transformers词典类对象、词典地址或词典名，用于实现文本分词和ID化
        max_seq_len (:obj:`int`): 预设的文本最大长度
    """
    def __init__(self, vocab, max_seq_len):
        super(SentenceTokenizer, self).__init__(vocab, max_seq_len)


if __name__ == '__main__':
    to = SentenceTokenizer('/Users/anulz/github/code/NLP/PTM/nezha-cn-base',30)
    print(to.max_seq_len)


