# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     sentence_cls_cus_dataset.py
   Description:
   Author:        GU Tianyi
   date:          2022/4/3
   desc:
-------------------------------------------------
"""
import jieba

from base._sentence_cls_dataset import SentenceClassificationDataset

class SentenceCustomizedClsDataset(SentenceClassificationDataset):
    """
    用于bert下句子分类任务
    """
    def __init__(self, *args, **kwargs):
        super(SentenceCustomizedClsDataset, self).__init__(*args, **kwargs)

    def _convert_to_custmoized_ids(self):
        pass
