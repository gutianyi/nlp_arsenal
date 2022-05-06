# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     sentence_cls_bert_dataset.py
   Description:
   Author:        GU Tianyi
   date:          2022/4/3
   desc:
-------------------------------------------------
"""
from dd_nlp_arsenal.dataset.base._sentence_cls_dataset import SentenceClassificationDataset

class SentenceBertClsDataset(SentenceClassificationDataset):
    """
    用于bert下句子分类任务
    """
    def __init__(self, *args, **kwargs):
        super(SentenceBertClsDataset, self).__init__(*args, **kwargs)

    def _convert_to_transformer_ids(self, bert_tokenizer):
        features = []
        for (index_, row_) in enumerate(self.dataset):
            encoding = bert_tokenizer.sequence_to_ids(row_['text'])

            feature = {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'ori_text': row_['text']
            }

            if self.is_train:
                label_ids = self.cat2id[row_['label']]
                feature['label_ids'] = label_ids

            features.append(feature)

        return features

