# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     _sentence_cls_dataset.py
   Description:
   Author:        GU Tianyi
   date:          2022/4/3
   desc:
-------------------------------------------------
"""
from dd_nlp_arsenal.dataset.base._dataset import BaseDataset

class SentenceClassificationDataset(BaseDataset):
    """
    用于句子分类任务
    """
    def _get_categories(self):
        self.categories = sorted(list(set([data['label'] for data in self.dataset])))

    def _convert_to_dataset(self, data_df):
        dataset = []

        data_df['text'] = data_df['text'].apply(lambda x: x.lower().strip())

        feature_names = list(data_df.columns)
        for index_, row_ in enumerate(data_df.itertuples()):
            dataset.append({
                feature_name_: getattr(row_, feature_name_)
                for feature_name_ in feature_names
            })

        self.dataset = dataset

if __name__ == '__main__':
    x = SentenceClassificationDataset('/Users/anulz/github/nlp_arsenal/test.csv')
    print(x.dataset)


