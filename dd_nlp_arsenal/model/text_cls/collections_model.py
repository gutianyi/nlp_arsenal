# -*- coding: utf-8 -*-
"""
//
//                       _oo0oo_
//                      o8888888o
//                      88" . "88
//                      (| -_- |)
//                      0\  =  /0
//                    ___/`---'\___
//                  .' \\|     |// '.
//                 / \\|||  :  |||// \
//                / _||||| -:- |||||- \
//               |   | \\\  -  /// |   |
//               | \_|  ''\---/''  |_/ |
//               \  .-\__  '-'  ___/-. /
//             ___'. .'  /--.--\  `. .'___
//          ."" '<  `.___\_<|>_/___.' >' "".
//         | | :  `- \`.;`\ _ /`;.`/ - ` : | |
//         \  \ `_.   \_ __\ /__ _/   .-` /  /
//     =====`-.____`.___ \_____/___.-`___.-'=====
//                       `=---='
//
//
//     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//               佛祖保佑         永无BUG
-------------------------------------------------
   File Name:     collections_model.py
   Description:
   Author:        GU Tianyi
   date:          2022/6/2
   desc:
-------------------------------------------------
"""
import torch
from torch import nn
from transformers import AutoConfig
from transformers import BertModel

from dd_nlp_arsenal.nn.base.basemodel import BasicModel
from dd_nlp_arsenal.nn.nezha.modeling import NeZhaModel
from dd_nlp_arsenal.nn.nezha.configuration import NeZhaConfig

MODEL_CLASSES = {
    'roberta': (AutoConfig, BertModel),
    'nazha': (NeZhaConfig, NeZhaModel),
}

class BertClsModel(BasicModel):

    def __init__(self, config):
        super(BertClsModel, self).__init__()
        model_config = MODEL_CLASSES[config.model_type][0].from_pretrained(config.bert_pretrained_name)
        model_config.update({'output_hidden_states': True})
        self.bert = MODEL_CLASSES[config.model_type][1].from_pretrained(config.bert_pretrained_name, config=model_config)
        self.out_layers = nn.Linear(model_config.hidden_size, config.n_classes)

        self.init_params([self.out_layers])

    def init_params(self, layers):
        for layer in layers:
            for m in layer.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight.data)
                    nn.init.constant_(m.bias.data, 0)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask)
        last_hidden_state = outputs[0]
        return self.out_layers(last_hidden_state[:,0]).squeeze()

class BertAttClsModel(BasicModel):
    def __init__(self, config):
        super().__init__()

        model_config = MODEL_CLASSES[config.model_type][0].from_pretrained(config.bert_pretrained_name)
        model_config.update({"output_hidden_states": True,
                       "hidden_dropout_prob": 0.0,
                       "layer_norm_eps": 1e-7})

        self.roberta = MODEL_CLASSES[config.model_type][1].from_pretrained(config.bert_pretrained_name, config=model_config)

        dim = self.roberta.pooler.dense.bias.shape[0]

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
        roberta_output = self.roberta(input_ids=input_ids,
                                      attention_mask=attention_mask)

        cls_outputs = torch.stack(
            [self.dropout(layer) for layer in roberta_output[2][-12:]], dim=0
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


class BertAttMeanClsModel(BasicModel):
    def __init__(self, config):
        super().__init__()

        model_config = MODEL_CLASSES[config.model_type][0].from_pretrained(config.bert_pretrained_name)
        self.bert = MODEL_CLASSES[config.model_type][1].from_pretrained(config.bert_pretrained_name, config=model_config)
        embedding_dim = self.bert.config.to_dict()['hidden_size']
        self.linear = nn.Linear(embedding_dim, config.n_classes)
        self._init_weights([self.linear], initializer_range=0.02)

    @staticmethod
    def _init_weights(blocks, **kwargs):
        """
        参数初始化，将 Linear / Embedding / LayerNorm 与 Bert 进行一样的初始化
        """
        for block in blocks:
            for module in block.modules():
                if isinstance(module, nn.Linear):
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Embedding):
                    nn.init.normal_(module.weight, mean=0, std=kwargs.pop('initializer_range', 0.02))
                elif isinstance(module, nn.LayerNorm):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask)
        last_hidden_state = outputs[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        logits = self.linear(mean_embeddings)
        return logits


if __name__ == '__main__':
    class Config():
        random_seed = 2022
        bert_pretrained_name = '/Users/anulz/github/code/NLP/PTM/bert-base-chinese'
        output_dir = '/'
        n_classes = 2

    config = Config()
    m = BertClsModel(config)
    print(m)

