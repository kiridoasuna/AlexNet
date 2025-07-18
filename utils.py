# -*- coding: UTF-8 -*-
'''
@Project ：AlexNet 
@File    ：utils.py
@Author  ：公众号：思维侣行
@Date    ：2025/7/17 09:49 
'''

import torch
import numpy as np
import random
import os

def set_random_seed(seed=42):
    """
    在初始化阶段设定随机种子
    :param seed:
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed) # 单GPU
        torch.cuda.manual_seed_all(seed) # 多GPU
        if hasattr(torch.backends, 'cudnn'):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            print("警告: torch.backends.cudnn 不可用，可能是由于PyTorch版本或CUDA支持问题。")

def set_sub_process_seed(work_id):
    """
    在DataLoader中给子进程设定随机种子
    :param work_id:
    :return:
    """
    sub_process_seed = torch.initial_seed() %2 ** 32
    np.random.seed(sub_process_seed)
    random.seed(sub_process_seed)

def set_main_process_seed(seed=42):
    """
    在使用DataLoader 洗牌时 给主进程设定随机种子
    :param seed:
    :return:
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return g

def get_train_and_val_data_size(total_size, p=0.8):
    """
    计算训练集和验证集的数据大小
    :param total_size: 数据总数
    :param p: 训练集占的比例
    :return: 训练集和验证集的大小
    """
    train_data_size = round(0.8 * total_size)
    val_data_size = total_size - train_data_size

    return train_data_size, val_data_size

if __name__ == '__main__':
    set_random_seed()