# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     base_task.py
   Description:
   Author:        GU Tianyi
   date:          2022/4/5
   desc:
-------------------------------------------------
"""
import torch.cuda
from dd_nlp_arsenal.factory.untils.ema import EMA
from dd_nlp_arsenal.factory.untils.tools import seed_torch, set_logger

import os
import logging

class BaseTask(object):
    """
    所有Task类的基类，封装Task类通用的方法和属性
    """
    def __init__(self, model, optimizer, loss_func, config):

        if config.multi_gpu:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            if torch.cuda.is_available():
                if config.cuda_device == -1:
                    self.device = torch.device("cuda")
                else:
                    self.device = torch.device(f"cuda:{config.cuda_device}")
                    # torch.cuda.set_device(config.device_id)
            else:
                self.device = torch.device("cpu")

        self.optimizer = optimizer
        self.model = model.to(self.device)
        self.loss_func = loss_func.to(self.device)
        self.scheduler = None

        seed_torch(config.seed)
        if config.multi_gpu:
            torch.cuda.manual_seed_all(config.seed)

        self.ema = EMA(self.model.parameters(), decay=config.ema_decay) if config.ema_decay else None

        # Set the logger
        try:
            if not config.save_log:
                config.save_log = True
        except:
            config.save_log = True

        self.logger, self.file_handler = set_logger(save=config.save_log, log_path=os.path.join(config.params_path, 'train.log'))
        logging.info(f"Model type: {config.pre_model_type}")
        logging.info("device: {}".format(self.device))

        logging.info('Init pre-train model...')

        config_str = '\nconfig is:\n'
        for item in dir(config):
            if "__" not in item:
                config_str += f"{item} = {config.__getattribute__(item)}\n"
        logging.info(config_str)

    def end_task(self, config):
        if config.save_log:
            self.logger.removeHandler(self.file_handler)
