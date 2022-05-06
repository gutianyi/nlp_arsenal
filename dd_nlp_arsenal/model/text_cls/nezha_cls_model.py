# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     nezha_model.py
   Description:
   Author:        GU Tianyi
   date:          2022/4/7
   desc:
-------------------------------------------------
"""
from torch import nn
import torch

from dd_nlp_arsenal.nn.base.basemodel import BasicModel
from dd_nlp_arsenal.nn.nezha.modeling import NeZhaModel
from dd_nlp_arsenal.nn.nezha.configuration import NeZhaConfig

class NezhaClsModel(BasicModel):
    """
    nezha cls模型
    """
    def __init__(self, config):
        super(NezhaClsModel, self).__init__()
        self.n_classes = config.n_classes
        # self.bert = NeZhaModel(config)
        self.bert = NeZhaModel.from_pretrained(config.bert_pretrained_name)
        self.classifier = nn.Linear(config.hidden_size, self.n_classes)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        # attention_mask = torch.ne(input_ids, 0)  #todo

        encoder_out, pooled_out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        logits = self.classifier(pooled_out)
        # outputs = (logits,) + (pooled_out,)

        # if labels is not None:
        #     loss_fct = nn.CrossEntropyLoss()
        #     # loss_fct = focal_loss()
        #     loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        #     outputs = (loss,) + outputs

        # return outputs
        return logits


class NezhaAttClsModel(BasicModel):
    def __init__(self, config):
        super().__init__()
        model_config = NeZhaConfig.from_pretrained(config.bert_pretrained_name)
        model_config.update({"output_hidden_states": True})
        self.nezha = NeZhaModel.from_pretrained(config.bert_pretrained_name, config=model_config)

        dim = self.nezha.pooler.dense.bias.shape[0]

        self.dropout = nn.Dropout(p=0.2)
        self.high_dropout = nn.Dropout(p=0.5)

        n_weights = 12
        weights_init = torch.zeros(n_weights).float()
        weights_init.data[:-1] = -3
        self.layer_weights = torch.nn.Parameter(weights_init)

        self.attention = nn.Sequential(
            nn.Linear(768, 768),
            nn.Tanh(),
            nn.Linear(768, 1),
            nn.Softmax(dim=1)
        )
        self.cls = nn.Sequential(
            nn.Linear(dim, config.n_classes)
        )
        self.init_params([self.cls, self.attention])

    def init_params(self, layers):
        for layer in layers:
            for m in layer.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight.data)
                    nn.init.constant_(m.bias.data, 0)


    def forward(self, input_ids, attention_mask):
        nezha_output = self.nezha(input_ids=input_ids,
                                      attention_mask=attention_mask)

        cls_outputs = torch.stack(
            [self.dropout(layer) for layer in nezha_output[2][-12:]], dim=0
        )
        cls_output = (
                torch.softmax(self.layer_weights, dim=0).unsqueeze(1).unsqueeze(1).unsqueeze(1) * cls_outputs).sum(
            0)

        logits = torch.mean(
            torch.stack(
                [torch.sum(self.attention(self.high_dropout(cls_output)) * cls_output, dim=1) for _ in range(5)],
                dim=0,
            ),
            dim=0,
        )
        return self.cls(logits)