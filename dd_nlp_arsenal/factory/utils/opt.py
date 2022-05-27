# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     opt.py
   Description:
   Author:        GU Tianyi
   date:          2022/4/25
   desc:
-------------------------------------------------
"""
from transformers import AdamW
def get_default_bert_optimizer(
    module,
    lr: float = 3e-5,
    out_lr: float = 2e-3,
    eps: float = 1e-6,
    correct_bias: bool = True,
    weight_decay: float = 1e-3,
):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        # pretrain model param
        # 衰减
        {"params": [p for n, p in module.named_parameters() if not any(nd in n for nd in no_decay) and "bert" in n],
            "weight_decay": weight_decay, "lr":lr},
        {"params": [p for n, p in module.named_parameters() if any(nd in n for nd in no_decay) and "bert" in n],
            "weight_decay": 0.0, "lr":lr},
        {'params': [p for n, p in module.named_parameters() if not any(nd in n for nd in no_decay) and'bert' not in n],
         'weight_decay': weight_decay, 'lr': out_lr},
        {'params': [p for n, p in module.named_parameters() if any(nd in n for nd in no_decay) and 'bert' not in n],
         'weight_decay': 0.0, 'lr': out_lr}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      eps=eps,
                      correct_bias=correct_bias,
                      weight_decay=weight_decay)
    return optimizer