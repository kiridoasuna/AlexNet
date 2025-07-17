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

if __name__ == '__main__':
    set_random_seed()