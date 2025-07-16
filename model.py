# -*- coding: UTF-8 -*-
'''
@Project ：AlexNet 
@File    ：model.py
@Author  ：公众号：思维侣行
@Date    ：2025/7/15 10:11 
'''
import torch
from torch import nn as nn
from torch.nn import ReLU, Conv2d, MaxPool2d, Linear, Flatten, Dropout
from torchsummary import summary

class AlexNet(nn.Module):
    def __init__(self):
        """
        Alex 模型的每一层的设定
        """
        super(AlexNet, self).__init__()
        self.c1 = Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4)
        self.s2 = MaxPool2d(kernel_size=3, stride=2)
        self.c3 = Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.s4 = MaxPool2d(kernel_size=3, stride=2)
        self.c5 = Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.c6 = Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.c7 = Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.s8 = MaxPool2d(kernel_size=3, stride=2)
        self.relu = ReLU()
        self.flatten = Flatten()
        self.f9 = Linear(in_features=256 * 6 * 6, out_features=4096)
        self.f10 = Linear(in_features=4096, out_features=4096)
        # 论文中的输出是1000，因为原论文中是为了解决1000个分类的问题。这回为了配合10个分类的数据集，而改为10
        self.f11 = Linear(in_features=4096, out_features=10)
        self.dropout = Dropout(p=0.5)

    def forward(self, x):
        """
        模型组装
        :param x: 输入的图片的数据
        :return:  输出的图片的分类(正常有1000个分类，这回为了配合数据集，设定了10个分类)
        """
        x = self.relu(self.c1(x))
        x = self.s2(x)
        x = self.relu(self.c3(x))
        x = self.s4(x)
        x = self.relu(self.c5(x))
        x = self.relu(self.c6(x))
        x = self.relu(self.c7(x))
        x = self.s8(x)
        x = self.flatten(x)
        x = self.f9(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.f10(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.f11(x)

        return x

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlexNet().to(device)
    # 本例使用灰度图，所以通道数是1
    print(summary(model, (1, 227, 227)))


