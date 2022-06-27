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
import logging

from tqdm import tqdm
from tqdm import trange
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

import torch
from torch import nn
from torch.utils.data import DataLoader

from transformers import get_linear_schedule_with_warmup

from dd_nlp_arsenal.factory.task.base_task import *
from dd_nlp_arsenal.factory.untils.tools import split_dataset
from dd_nlp_arsenal.factory.untils.cus_loss import focal_loss, compute_kl_loss
from dd_nlp_arsenal.factory.untils.attack import FGM, PGD, AWP

METRICS_MAP = {'acc': accuracy_score,
               'f1': f1_score}

class SentenceCLSTask(BaseTask):
    """
    单句文本分类任务
    """

    def __init__(self, *args, **kwargs):
        super(SentenceCLSTask, self).__init__(*args, **kwargs)

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
        )

        if not is_full_data:
            val_loader = DataLoader(
                val_data,
                batch_size=config.val_batch_size,
                shuffle=False,
                num_workers=config.num_workers,
            )

        if self.scheduler is None and config.scheduler_type is not None:
            if config.scheduler_type == 'get_linear_schedule_with_warmup':
                total_steps = len(train_loader) * config.n_epoch
                self.scheduler = get_linear_schedule_with_warmup(
                    self.optimizer,
                    num_warmup_steps=0,
                    num_training_steps=total_steps
                )

        if config.attack_func == 'fgm':
            attack_func = FGM(self.model)
        elif config.attack_func == 'pgd':
            attack_func = PGD(self.model)
        elif config.attack_func == 'awp':
            attack_func = AWP(self.model)
        else:
            attack_func = None

        best_f1 = float('-inf')
        for epoch in range(config.n_epoch):
            logging.info(f'Epoch {epoch + 1}/{config.n_epoch}')
            logging.info('-' * 10)
            train_loss, train_f1 = self._train_epoch(self.model, train_loader, attack_func, config)
            logging.info(f'Train loss {train_loss} train {config.metrics} {train_f1}')

            if not is_full_data:
                val_loss, val_f1 = self._eval_model(self.model, val_loader, config)
                logging.info(f'Val   loss {val_loss}  {config.metrics} {val_f1}')

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

        self.end_task(config)

    def _train_epoch(self, model, data_loader, attack_func, config):
        model = model.train()
        losses = []
        predictions = []
        real_values = []
        tqdm_t = trange(len(data_loader), ascii=True)
        for _step, _ in enumerate(tqdm_t):
            d = next(iter(data_loader))

            input_ids = d["input_ids"].to(self.device)
            attention_mask = d["attention_mask"].to(self.device)
            targets = d["label_ids"].to(self.device)

            self.optimizer.zero_grad()

            if config.is_use_rdrop:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                outputs2 = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                # cross entropy loss for classifier
                ce_loss = 0.5 * (self.loss_func(outputs, targets) + self.loss_func(outputs2, targets))
                kl_loss = compute_kl_loss(outputs, outputs2, is_mean=config.rdrop_ismean)
                # carefully choose hyper-parameters
                loss = ce_loss + config.alpha * kl_loss
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                loss = self.loss_func(outputs, targets)
            _, preds = torch.max(outputs, dim=1)

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
            elif config.attack_func == 'pgd' or config.attack_func == 'awp':
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

            predictions.extend(preds)
            real_values.extend(targets)

        if config.is_use_swa:
            self.optimizer.swap_swa_sgd()

        predictions = torch.stack(predictions).cpu()
        real_values = torch.stack(real_values).cpu()
        
        metrics_out = METRICS_MAP[config.metrics](real_values, predictions, average='macro') if config.metrics != 'acc' else  METRICS_MAP[config.metrics](real_values, predictions)

        return np.mean(losses), metrics_out

    def _eval_model(self, model, data_loader, config):
        model = model.eval()  # 验证预测模式
        if self.ema is not None:
            # eval前，进行ema的权重替换；eval之后，恢复原来模型的参数
            self.ema.store(self.model.parameters())
            self.ema.copy_to(self.model.parameters())

        losses = []
        predictions = []
        real_values = []

        with torch.no_grad():
            for d in data_loader:
                input_ids = d["input_ids"].to(self.device)
                attention_mask = d["attention_mask"].to(self.device)
                targets = d["label_ids"].to(self.device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                _, preds = torch.max(outputs, dim=1)

                loss = self.loss_func(outputs, targets)
                losses.append(loss.item())
                predictions.extend(preds)
                real_values.extend(targets)

        if config.is_use_swa:
            self.optimizer.swap_swa_sgd()

        if self.ema is not None:
            self.ema.restore(self.model.parameters())

        predictions = torch.stack(predictions).cpu()
        real_values = torch.stack(real_values).cpu()

        metrics_out = METRICS_MAP[config.metrics](real_values, predictions, average='macro') if config.metrics != 'acc' else  METRICS_MAP[config.metrics](real_values, predictions)

        return np.mean(losses), metrics_out
