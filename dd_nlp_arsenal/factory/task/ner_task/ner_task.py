# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     ner_task.py
   Description:
   Author:        GU Tianyi
   date:          2022/4/5
   desc:
-------------------------------------------------
"""

from tqdm import trange
import numpy as np
from sklearn.metrics import accuracy_score

import torch
from torch import nn
from torch.utils.data import DataLoader

from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import get_linear_schedule_with_warmup

from dd_nlp_arsenal.factory.task.base_task import *
from dd_nlp_arsenal.factory.untils.tools import split_dataset
from dd_nlp_arsenal.factory.untils.cus_loss import focal_loss, compute_kl_loss
from dd_nlp_arsenal.factory.untils.attack import FGM, PGD

def global_pointer_f1_score(y_true, y_pred):
    y_pred = torch.gt(y_pred, 0)
    return torch.sum(y_true * y_pred).item(), torch.sum(y_true + y_pred).item()

class GlobalPointerNERTask(BaseTask):
    """
    GlobalPointer NER Task
    """
    def __init__(self, *args, **kwargs):
        super(GlobalPointerNERTask, self).__init__(*args, **kwargs)

    def fit(self, train_data, config, val_data=None, is_full_data=False):
        """
        训练func包装器
        :param train_data: 训练集 封装的dataset类
        :param config: 参数文件map
        :param val_data: 验证集
        :param optimizer: 优化器
        """
        if not is_full_data:
            if val_data is None:
                train_data, val_data = split_dataset(train_data.dataset, seed=config.seed)
            else:
                train_data, val_data = train_data.dataset, val_data.dataset

        train_loader = DataLoader(
            train_data,
            batch_size=config.train_batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            # collate_fn=self._train_collate_fn
            )
        if not is_full_data:
            val_loader = DataLoader(
                val_data,
                batch_size=config.val_batch_size,
                shuffle=False,
                num_workers=config.num_workers,
                # collate_fn=self._train_collate_fn
                )

        if self.scheduler is None and config.scheduler_type is not None:
            if config.scheduler_type == 'get_linear_schedule_with_warmup':
                total_steps = len(train_loader) * config.n_epoch
                self.scheduler = get_linear_schedule_with_warmup(
                    self.optimizer,
                    num_warmup_steps=0,
                    num_training_steps=total_steps
                )

        """ignored_params = list(map(id, self.model.out_layers.parameters()))  # 返回的是parameters的 内存地址
        base_params = filter(lambda p: id(p) not in ignored_params, self.model.parameters())
        optimizer = AdamW([
            {'params': base_params},
            # {'params': model.attention.parameters(), 'lr': 1e-4},
            {'params': self.model.out_layers.parameters(), 'lr': 1e-4}], lr=config.lr, correct_bias=False, betas=(0.9, 0.98))"""

        if config.attack_func == 'fgm':
            attack_func = FGM(self.model)
        elif config.attack_func == 'pgd':
            attack_func = PGD(self.model)
        else:
            attack_func = None

        best_f1 = float('-inf')
        for epoch in range(config.n_epoch):
            logging.info(f'Epoch {epoch + 1}/{config.n_epoch}')
            logging.info('-' * 10)
            train_loss = self._train_epoch(self.model, train_loader, attack_func, config)
            logging.info(f'Train loss {train_loss}')

            if not is_full_data:
                val_loss, val_p, val_r, val_f1 = self._eval_model(self.model, val_loader, config)
                logging.info(f'Val   loss {val_loss}  f1_score {val_f1}')

                if epoch + 1 >= config.min_store_epoch:
                    if val_f1 > best_f1:
                        logging.info(f'saving best_model_state val loss is {val_loss}...')
                        if config.trained_model_path is not None:
                            self.model.save(name=config.trained_model_path)
                        else:
                            self.model.save()
                        best_f1 = val_f1
            else:
                logging.info(f'saving best_model_state val loss is {train_loss}...')
                if self.ema is not None:
                    # eval前，进行ema的权重替换；eval之后，恢复原来模型的参数
                    self.ema.store(self.model.parameters())
                    self.ema.copy_to(self.model.parameters())

                if config.trained_model_path is not None:
                    self.model.save(name=config.trained_model_path)
                else:
                    self.model.save()

            if self.ema is not None:
                self.ema.restore(self.model.parameters())


    def _train_epoch(self, model, data_loader, attack_func, config):
        model = model.train()
        losses = []
        tqdm_t = trange(len(data_loader), ascii=True)
        for _step, _ in enumerate(tqdm_t):
            d = next(iter(data_loader))

            input_ids = d["input_ids"].to(self.device)
            attention_mask = d["attention_mask"].to(self.device)
            token_type_ids = d["token_type_ids"].to(self.device)
            targets = d["label_ids"].to(self.device)

            self.optimizer.zero_grad()

            if config.is_use_rdrop:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
                outputs2 = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
                # cross entropy loss for classifier
                ce_loss = 0.5 * (self.loss_func(outputs, targets) + self.loss_func(outputs2, targets))
                kl_loss = compute_kl_loss(outputs, outputs2)
                # carefully choose hyper-parameters
                loss = ce_loss + config.alpha * kl_loss
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
                loss = self.loss_func(outputs, targets)

            losses.append(loss.item())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            if config.attack_func == 'fgm':
                # 对抗训练
                attack_func.attack()  # 在embedding上添加对抗扰动
                # optimizer.zero_grad() # 如果不想累加梯度，就把这里的注释取消
                loss_adv = self.loss_func(model(input_ids=input_ids,
                                                attention_mask=attention_mask), targets)
                loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                attack_func.restore()  # 恢复embedding参数
            elif config.attack_func == 'pgd':
                attack_func.backup_grad()
                # 对抗训练
                for t in range(config.pgd_k):
                    attack_func.attack(is_first_attack=(t == 0))  # 在embedding上添加对抗扰动, first attack时备份param.data
                    if t != config.pgd_k - 1:
                        self.optimizer.zero_grad()
                    else:
                        attack_func.restore_grad()
                    loss_adv = self.loss_func(model(input_ids=input_ids,
                                                    attention_mask=attention_mask), targets)
                    loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                attack_func.restore()  # 恢复embedding参数

            self.optimizer.step()
            if self.ema is not None:
                self.ema.update(self.model.parameters())

            if config.is_use_swa:
                if _step > 10 and _step % 5 == 0:
                    # self.optimizer.update_swa()
                    self.optimizer.update_swa_group(self.optimizer.param_groups[1])
            if self.scheduler is not None:
                self.scheduler.step()
            tqdm_t.set_postfix(loss='{:05.3f}'.format(loss.item()))

        if config.is_use_swa:
            self.optimizer.swap_swa_sgd()

        return np.mean(losses)

    def _eval_model(self, model, data_loader, config):
        model = model.eval()  # 验证预测模式
        if self.ema is not None:
            # eval前，进行ema的权重替换；eval之后，恢复原来模型的参数
            self.ema.store(self.model.parameters())
            self.ema.copy_to(self.model.parameters())

        losses = []
        precision, recall = 0, 0

        with torch.no_grad():
            for d in data_loader:
                input_ids = d["input_ids"].to(self.device)
                attention_mask = d["attention_mask"].to(self.device)
                token_type_ids = d["token_type_ids"].to(self.device)
                targets = d["label_ids"].to(self.device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )

                loss = self.loss_func(outputs, targets)
                losses.append(loss.item())

                P, R = global_pointer_f1_score(
                    targets.to_dense().cpu(),
                    outputs.cpu()
                )
                precision += P
                recall += R

        if config.is_use_swa:
            self.optimizer.swap_swa_sgd()

        if self.ema is not None:
            self.ema.restore(self.model.parameters())
        return np.mean(losses), precision, recall, 2*precision/recall