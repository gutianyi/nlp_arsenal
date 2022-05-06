#                       _oo0oo_
#                      o8888888o
#                      88" . "88
#                      (| -_- |)
#                      0\  =  /0
#                    ___/`---'\___
#                  .' \\|     |// '.
#                 / \\|||  :  |||// \
#                / _||||| -:- |||||- \
#               |   | \\\  -  /// |   |
#               | \_|  ''\---/''  |_/ |
#               \  .-\__  '-'  ___/-. /
#             ___'. .'  /--.--\  `. .'___
#          ."" '<  `.___\_<|>_/___.' >' "".
#         | | :  `- \`.;`\ _ /`;.`/ - ` : | |
#         \  \ `_.   \_ __\ /__ _/   .-` /  /
#     =====`-.____`.___ \_____/___.-`___.-'=====
#                       `=---='
#
#                -*- coding: utf-8 -*-
#     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#               佛祖保佑         永无BUG
# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/5/4 18:06
# @Author  : GU Tianyi
# @File    : crf_bert.py

from transformers import AutoConfig
from transformers import AutoModel

import torch
from torch import nn
from dd_nlp_arsenal.nn.base.basemodel import BasicModel
from dd_nlp_arsenal.nn.layer.crf_block import CRFLayer
from dd_nlp_arsenal.nn.nezha.modeling import NeZhaModel
from dd_nlp_arsenal.nn.layer.downs_encoder import BiLSTM, RTransformer, TENER
from dd_nlp_arsenal.model.ner.crf_bert.utils import Params

class CrfBert(BasicModel):
    """
    crf bert model
    适配各种下游模型
    """
    def __init__(self, config, ds_encoder_type='LSTM'):
        super(CrfBert, self).__init__()
        params = Params(pre_model_type=config.pre_model_type, ex_index=0, fold_id=0)
        self.num_labels = len(params.tags)

        # pretrain model
        self.pre_model_type = params.pre_model_type
        if self.pre_model_type == 'NEZHA':
            self.bert = NeZhaModel.from_pretrained(config.bert_pretrained_name)
        elif self.pre_model_type == 'RoBERTa':
            model_config = AutoConfig.from_pretrained(config.bert_pretrained_name)
            self.bert = AutoModel.from_pretrained(config.bert_pretrained_name, config=model_config)
        else:
            raise ValueError('Pre-train Model type must be NEZHA or ELECTRA or RoBERTa!')

        # 动态权重
        self.fusion_layers = params.fusion_layers
        self.dym_weight = nn.Parameter(torch.ones((self.fusion_layers, 1, 1, 1)),
                                       requires_grad=True)

        # downstream encoder
        self.ds_encoder = ds_encoder_type
        if self.ds_encoder == 'LSTM':
            self.bilstm = BiLSTM(self.num_labels, embedding_size=config.hidden_size, hidden_size=params.lstm_hidden,
                                 num_layers=params.lstm_layer,
                                 dropout=params.drop_prob, with_ln=True)
        elif self.ds_encoder == 'TENER':
            self.tener = TENER(tag_size=self.num_labels, embed_size=config.hidden_size, dropout=params.drop_prob,
                               num_layers=params.num_layers, d_model=params.tener_hs, n_head=params.num_heads)
        elif self.ds_encoder == 'RTRANS':
            self.rtrans = RTransformer(tag_size=self.num_labels, dropout=params.drop_prob, d_model=config.hidden_size,
                                       ksize=params.k_size, h=params.rtrans_heads)
        else:
            raise ValueError('Downstream encoder type must be LSTM or TENER or RTRANS!')

        # crf
        self.crf = CRFLayer(self.num_labels, params)

        # init weights
        self.init_weights()
        self.reset_params()

    def reset_params(self):
        nn.init.xavier_normal_(self.dym_weight)

    def get_dym_layer(self, outputs):
        """获取动态权重融合后的BERT output.
        Args:
            outputs: origin bert output
        Returns:
            sequence_output: 融合后的bert encoder output. (bs, seq_len, hs)
        """
        if self.pre_model_type in ('ELECTRA', 'NEZHA'):
            fusion_idx = 0
        elif self.pre_model_type == 'RoBERTa':
            fusion_idx = 2
        else:
            raise ValueError('Pre-train Model type must be NEZHA or ELECTRA or RoBERTa!')

        hidden_stack = torch.stack(outputs[fusion_idx][-self.fusion_layers:],
                                   dim=0)  # (num_layers, bs, seq_len, hs)
        sequence_output = torch.sum(hidden_stack * self.dym_weight,
                                    dim=0)  # (bs, seq_len, hs)
        return sequence_output

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            labels=None,
    ):
        """
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: 各元素的值为0或1，避免在padding的token上计算attention。(batch_size, seq_len)
            token_type_ids: 就是token对应的句子类型id，值为0或1。为空自动生成全0。(batch_size, seq_len)
            labels: (batch_size, seq_len)
        """
        # pre-train model
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True
        )  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[-1][-1]
        # fusion BERT layers
        # sequence_output = self.get_dym_layer(outputs)  # (batch_size, seq_len, hidden_size[embedding_dim])
        batch_size, seq_len, tag_size = sequence_output.size()

        if self.ds_encoder == 'LSTM':
            # (seq_len, batch_size, tag_size)
            feats = self.bilstm.get_lstm_features(sequence_output.transpose(1, 0), attention_mask.transpose(1, 0))
        elif self.ds_encoder == 'TENER':
            feats = self.tener(sequence_output, attention_mask)  # (bs, seq_len, tag_size)
            feats = feats.transpose(1, 0)
        elif self.ds_encoder == 'RTRANS':
            feats = self.rtrans(sequence_output, attention_mask)
            feats = feats.transpose(1, 0)
        else:
            raise TypeError('Downstream encoder type must be LSTM or TENER or RTRANS!')

        if labels is not None:
            # CRF
            # total scores
            forward_score = self.crf(feats, attention_mask.transpose(1, 0))
            gold_score = self.crf.score_sentence(feats, labels.transpose(1, 0),
                                                 attention_mask.transpose(1, 0))
            loss = (forward_score - gold_score).sum() / batch_size
            return loss
        else:
            # 维特比算法
            best_paths = self.crf.viterbi_decode(feats, attention_mask.transpose(1, 0))
            return best_paths

