# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     _dataset.py
   Description:
   Author:        GU Tianyi
   date:          2022/4/3
   desc:
-------------------------------------------------
"""
import pandas as pd

from torch.utils.data import Dataset
from pandas.core.frame import DataFrame

class BaseDataset(Dataset):
    """
    Dataset base类
    """

    def __init__(self, data, is_train=True, categories=None):
        """
        BaseDataset 参数
        :param data: DataFrame or string
                数据或者数据地址
        :param is_train: bool, default=None
                数据集是否为训练集数据
        :param categories: list, default=None
                数据类别
        """
        self.is_train = is_train
        self.dataset = None

        if isinstance(data, DataFrame):
            if 'label' in data.columns:
                data['label'] = data['label'].astype(str)
            self._convert_to_dataset(data)
        else:
            self._load_dataset(data)
        if categories:
            self.categories = categories
        else:
            self._get_categories()

        if self.categories is not None:
            self.cat2id = dict(zip(self.categories, range(len(self.categories))))
            self.id2cat = dict(zip(range(len(self.categories)), self.categories))

            self.class_num = len(self.cat2id)



    def _load_dataset(self, data_path):
        """
        加载数据--->dataset
        :param data_path:
        :return:
        """
        if isinstance(data_path, str):
            data_format = data_path.split('.')[-1]
            data_df = None
            if data_format == 'csv':
                data_df = pd.read_csv(data_path, dtype={'label': str})
            elif data_format == 'txt:':
                data_df = pd.read_table(data_path, sep='\t', dtype={'label': str})
            elif data_format == 'tsv':
                data_df = pd.read_csv(data_path, sep='\t', dtype={'label': str})
            else:
                ValueError('The format of data file should be one of these(csv, txt, tsv)!')
        else:
            ValueError('The data_path should be string type!')

        self._convert_to_dataset(data_df)


    def _convert_to_dataset(self, dataset):
        """
        dataframe--->dataset
        :param dataset:
        :return:
        """
        raise NotImplementedError()

    def convert_to_ids(self, tokenizer=None):
        """
        实例化后需要被调用的function
        将文本转化成id的形式
        :param tokenizer:编码器
        :return:
        """
        if tokenizer.tokenizer_type == 'transformer':
            features = self._convert_to_transformer_ids(tokenizer)
        else:
            features = self._convert_to_custmoized_ids()

        self.dataset = features

    def _convert_to_transformer_ids(self, bert_tokenizer):
        pass

    def _convert_to_custmoized_ids(self):
        pass

    def _get_categories(self):
        """
        得到种类标签
        :return:
        """
        raise NotImplementedError()

    def __getitem__(self, index):
        """
        torch dataset要求def
        :param item:
        :return:
        """
        return self.dataset[index]

    def __len__(self):
        """
        torch dataset要求def
        :return:
        """
        return len(self.dataset)
