# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     config.py
   Description:
   Author:        GU Tianyi
   date:          2022/4/5
   desc:
-------------------------------------------------
"""
import os
from itertools import chain
import json
from pathlib import Path

import torch

PreModelDir = {
    "NEZHA": 'nezha-cn-base',
    "RoBERTa": 'chinese-roberta-wwm-ext'
}

class Config:
    """参数定义
    """

    def __init__(self, pre_model_type='NEZHA', ex_index=1, fold_id=0):
        # 根路径
        self.root_path = Path(os.path.abspath(os.path.dirname(__file__)))
        # 数据集路径
        self.data_dir = self.root_path.parent / f'data/fold{fold_id}'
        # 参数路径
        self.params_path = self.root_path / f'experiments/ex{ex_index}/fold{fold_id}'
        # 模型保存路径
        self.model_dir = self.root_path / f'model/ex{ex_index}/fold{fold_id}'

        self.pre_model_type = pre_model_type
        # downstream encoder type
        self.ds_encoder_type = 'LSTM'
        # 预训练模型路径
        self.bert_model_dir = f'/home/hadoop-grocery-rc/workdir/gty/model_data/bert/{PreModelDir[self.pre_model_type]}'
        # device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_gpu = torch.cuda.device_count()
        # 标签列表

        # 读取保存的data
        self.data_cache = True
        self.train_batch_size = 256
        self.val_batch_size = 256
        self.test_batch_size = 256

        # patience strategy
        # 最小训练次数
        self.min_epoch_num = 3
        # 容纳的提高量(f1-score)
        self.patience = 0.05
        # 容纳多少次未提高
        self.patience_num = 5

        self.seed = 2022
        # 句子最大长度(pad)
        self.max_seq_length = 110
        # 是否使用fgm
        self.is_use_fgm = False

        # BERT多层融合
        self.fusion_layers = 4
        # learning_rate
        self.fin_tuning_lr = 2e-5
        self.downs_en_lr = 1e-4
        self.crf_lr = self.fin_tuning_lr * 1000
        # 梯度截断
        self.clip_grad = 2.
        # dropout prob
        self.drop_prob = 0.3
        # 权重衰减系数
        self.weight_decay_rate = 0.01
        self.warmup_prop = 0.1
        self.gradient_accumulation_steps = 2

        # lstm hidden size
        self.lstm_hidden = 256
        # lstm layer num
        self.lstm_layer = 1

        # tener layers
        self.num_layers = 1
        # tener hidden size
        self.tener_hs = 256
        # tener head num
        self.num_heads = 4

        # rtrans
        self.k_size, self.rtrans_heads = 10, 4

    def get(self):
        """Gives dict-like access to Params instance by `params.show['learning_rate']"""
        return self.__dict__

    def load(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        """保存配置到json文件
        """
        params = {}
        with open(json_path, 'w') as f:
            for k, v in self.__dict__.items():
                if isinstance(v, (str, int, float, bool)):
                    params[k] = v
            json.dump(params, f, indent=4)