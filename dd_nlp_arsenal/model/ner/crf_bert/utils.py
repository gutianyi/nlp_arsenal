# /usr/bin/env python
# coding=utf-8
"""utils"""
import logging
import os
import shutil
from itertools import chain
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.init as init

# 定义实体标注
EN_DICT = {'类别1': ['B-1', 'I-1'],
 '类别2': ['B-2', 'I-2'],
 '类别3': ['B-3', 'I-3'],
 '类别4': ['B-4', 'I-4'],
 '类别5': ['B-5', 'I-5'],
 '类别6': ['B-6', 'I-6'],
 '类别7': ['B-7', 'I-7'],
 '类别8': ['B-8', 'I-8'],
 '类别9': ['B-9', 'I-9'],
 '类别10': ['B-10', 'I-10'],
 '类别11': ['B-11', 'I-11'],
 '类别12': ['B-12', 'I-12'],
 '类别13': ['B-13', 'I-13'],
 '类别14': ['B-14', 'I-14'],
 '类别15': ['B-15', 'I-15'],
 '类别16': ['B-16', 'I-16'],
 '类别17': ['B-17', 'I-17'],
 '类别18': ['B-18', 'I-18'],
 '类别19': ['B-19', 'I-19'],
 '类别20': ['B-20', 'I-20'],
 '类别21': ['B-21', 'I-21'],
 '类别22': ['B-22', 'I-22'],
 '类别23': ['B-23', 'I-23'],
 '类别24': ['B-24', 'I-24'],
 '类别25': ['B-25', 'I-25'],
 '类别26': ['B-26', 'I-26'],
 '类别27': ['B-27', 'I-27'],
 '类别28': ['B-28', 'I-28'],
 '类别29': ['B-29', 'I-29'],
 '类别30': ['B-30', 'I-30'],
 '类别31': ['B-31', 'I-31'],
 '类别32': ['B-32', 'I-32'],
 '类别33': ['B-33', 'I-33'],
 '类别34': ['B-34', 'I-34'],
 '类别35': ['B-35', 'I-35'],
 '类别36': ['B-36', 'I-36'],
 '类别37': ['B-37', 'I-37'],
 '类别38': ['B-38', 'I-38'],
 '类别39': ['B-39', 'I-39'],
 '类别40': ['B-40', 'I-40'],
 '类别41': ['B-41', 'I-41'],
 '类别42': ['B-42', 'I-42'],
 '类别43': ['B-43', 'I-43'],
 '类别44': ['B-44', 'I-44'],
 '类别45': ['B-45', 'I-45'],
 '类别46': ['B-46', 'I-46'],
 '类别47': ['B-47', 'I-47'],
 '类别48': ['B-48', 'I-48'],
 '类别49': ['B-49', 'I-49'],
 '类别50': ['B-50', 'I-50'],
 '类别51': ['B-51', 'I-51'],
 '类别52': ['B-52', 'I-52'],
 '类别53': ['B-53', 'I-53'],
 '类别54': ['B-54', 'I-54'],
 'Others': 'O'}

IO2STR = {'1': '类别1',
 '2': '类别2',
 '3': '类别3',
 '4': '类别4',
 '5': '类别5',
 '6': '类别6',
 '7': '类别7',
 '8': '类别8',
 '9': '类别9',
 '10': '类别10',
 '11': '类别11',
 '12': '类别12',
 '13': '类别13',
 '14': '类别14',
 '15': '类别15',
 '16': '类别16',
 '17': '类别17',
 '18': '类别18',
 '19': '类别19',
 '20': '类别20',
 '21': '类别21',
 '22': '类别22',
 '23': '类别23',
 '24': '类别24',
 '25': '类别25',
 '26': '类别26',
 # '27': '类别27',
 '28': '类别28',
 '29': '类别29',
 '30': '类别30',
 '31': '类别31',
 '32': '类别32',
 '33': '类别33',
 '34': '类别34',
 '35': '类别35',
 '36': '类别36',
 '37': '类别37',
 '38': '类别38',
 '39': '类别39',
 '40': '类别40',
 '41': '类别41',
 '42': '类别42',
 '43': '类别43',
 '44': '类别44',
 # '45': '类别45',
 '46': '类别46',
 '47': '类别47',
 '48': '类别48',
 '49': '类别49',
 '50': '类别50',
 '51': '类别51',
 '52': '类别52',
 '53': '类别53',
 '54': '类别54'}
STR2IO = {v: k for k, v in IO2STR.items()}

PreModelDir = {
    "NEZHA": 'nezha-cn-base',
    "RoBERTa": 'chinese-roberta-wwm-ext'
}


class Params:
    """参数定义
    """

    def __init__(self, pre_model_type='NEZHA', ex_index=1, fold_id=0):
        # 根路径
        self.root_path = Path(os.path.abspath(os.path.dirname(__file__)))
        # 数据集路径
        # self.data_dir = self.root_path / f'data/fold{fold_id}'
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
        self.tags = list(chain(*EN_DICT.values()))
        # 用于CRF的标签
        self.tags.extend(["<START_TAG>", "<END_TAG>"])

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


class RunningAverage:
    """A simple class that maintains the running average of a quantity
    记录平均损失

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)


def set_logger(save=False, log_path=None):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if save and not os.path.exists(os.path.dirname(log_path)):
        os.makedirs(os.path.dirname(log_path))

    if not logger.handlers:
        if save:
            # Logging to a file
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
            logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_checkpoint(state, is_best, checkpoint):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'

    Args:
        state: (dict) contains the entire model, may contain other keys such as epoch, optimizer
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.makedirs(checkpoint)
    torch.save(state, filepath)
    # 如果是最好的checkpoint则以best为文件名保存
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))


def load_checkpoint(checkpoint, optimizer=True):
    """Loads entire model from file_path. If optimizer is True, loads
    optimizer assuming it is present in checkpoint.

    Args:
        checkpoint: (string) filename which needs to be loaded
        optimizer: (bool) resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise ValueError("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))

    if optimizer:
        return checkpoint['model'], checkpoint['optim']
    return checkpoint['model']


def initial_parameter(net, initial_method=None):
    r"""A method used to initialize the weights of PyTorch models.

    :param net: a PyTorch model or a List of Pytorch model
    :param str initial_method: one of the following initializations.

            - xavier_uniform
            - xavier_normal (default)
            - kaiming_normal, or msra
            - kaiming_uniform
            - orthogonal
            - sparse
            - normal
            - uniform

    """
    if initial_method == 'xavier_uniform':
        init_method = init.xavier_uniform_
    elif initial_method == 'xavier_normal':
        init_method = init.xavier_normal_
    elif initial_method == 'kaiming_normal' or initial_method == 'msra':
        init_method = init.kaiming_normal_
    elif initial_method == 'kaiming_uniform':
        init_method = init.kaiming_uniform_
    elif initial_method == 'orthogonal':
        init_method = init.orthogonal_
    elif initial_method == 'sparse':
        init_method = init.sparse_
    elif initial_method == 'normal':
        init_method = init.normal_
    elif initial_method == 'uniform':
        init_method = init.uniform_
    else:
        init_method = init.xavier_normal_

    def weights_init(m):
        # classname = m.__class__.__name__
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv3d):  # for all the cnn
            if initial_method is not None:
                init_method(m.weight.data)
            else:
                init.xavier_normal_(m.weight.data)
            init.normal_(m.bias.data)
        elif isinstance(m, nn.LSTM):
            for w in m.parameters():
                if len(w.data.size()) > 1:
                    init_method(w.data)  # weight
                else:
                    init.normal_(w.data)  # bias
        elif m is not None and hasattr(m, 'weight') and \
                hasattr(m.weight, "requires_grad"):
            if len(m.weight.size()) > 1:
                init_method(m.weight.data)
            else:
                init.normal_(m.weight.data)
        else:
            for w in m.parameters():
                if w.requires_grad:
                    if len(w.data.size()) > 1:
                        init_method(w.data)  # weight
                    else:
                        init.normal_(w.data)  # bias
                # print("init else")

    if isinstance(net, list):
        for n in net:
            n.apply(weights_init)
    else:
        net.apply(weights_init)


class FGM:
    """扰动训练(Fast Gradient Method)"""

    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='embeddings.'):
        """在embedding层中加扰动
        :param epsilon: 系数
        :param emb_name: 模型中embedding的参数名
        """
        #
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name and 'LayerNorm' not in name:
                # 保存原始参数
                self.backup[name] = param.data.clone()
                # scale
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    # 扰动因子
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='embeddings.'):
        """恢复扰动前的参数
        :param emb_name: 模型中embedding的参数名
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name and 'LayerNorm' not in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class PGD:
    """扰动训练(Projected Gradient Descent)"""

    def __init__(self, model):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, epsilon=1., alpha=0.3, emb_name='emb.', is_first_attack=False):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)

    def restore(self, emb_name='emb.'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = self.grad_backup[name]
