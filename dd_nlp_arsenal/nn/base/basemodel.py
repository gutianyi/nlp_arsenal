# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     basemodel.py
   Description:
   Author:        GU Tianyi
   date:          2022/4/4
   desc:
-------------------------------------------------
"""
import logging
import os
import torch
import time

class BasicModel(torch.nn.Module):
    def __init__(self):
        """
        封装了nn.Module，主要是提供了save和load两个方法
        """
        super(BasicModel, self).__init__()
        self.model_name = str(self.__class__.__name__)

    def load(self, path):
        """
        可加载指定路径的模型
        :param path: str (指定路径)
        :return:
        """
        self.load_state_dict(torch.load(path))

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        :param name: str (模型参数)
        :return:
        """
        if name is None:
            if not os.path.exists('./checkpoints'):
                os.makedirs('./checkpoints')
            prefix = 'checkpoints/' + self.model_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(), name)