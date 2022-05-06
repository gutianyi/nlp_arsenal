# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     global_pointer_bert.py
   Description:
   Author:        GU Tianyi
   date:          2022/4/9
   desc:
-------------------------------------------------
"""
from transformers import AutoConfig
from transformers import AutoModel
from torch import nn
from dd_nlp_arsenal.nn.base.basemodel import BasicModel
from dd_nlp_arsenal.nn.layer.global_pointer_block import GlobalPointer


class GlobalPointerBert(BasicModel):
    """
    GlobalPointer + Bert 的命名实体模型

    Args:
        config: 模型的配置对象
        bert_trained (:obj:`bool`, optional): 预训练模型的参数是否可训练

    Reference:
        [1] https://www.kexue.fm/archives/8373
    """
    def __init__(self, config, encoder_trained=True, head_size=64):
        super(GlobalPointerBert, self).__init__()
        self.num_labels = config.num_labels

        model_config = AutoConfig.from_pretrained(config.bert_pretrained_name)
        self.bert = AutoModel.from_pretrained(config.bert_pretrained_name, config=model_config)

        for param in self.bert.parameters():
            param.requires_grad = encoder_trained

        self.global_pointer = GlobalPointer(
            self.num_labels,
            head_size,
            config.hidden_size
        )

        # self.init_params([self.global_pointer])

    def init_params(self, layers):
        for layer in layers:
            for m in layer.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight.data)
                    nn.init.constant_(m.bias.data, 0)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        **kwargs
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
            output_hidden_states=True
        ).hidden_states
        sequence_output = outputs[-1]

        logits = self.global_pointer(sequence_output, mask=attention_mask)

        return logits