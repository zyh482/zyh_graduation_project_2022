# -*- coding:utf-8 -*-
# CREATED BY: zhangyuhan
# CREATED ON: 2022/3/29 3:11 PM
# LAST MODIFIED ON:
# AIM:
"""Dataset and Dataloader for project-training"""

import torch.utils.data


class ProjectBatchDataset(torch.utils.data.Dataset):
    def __init__(self, bert_input_list=[], bias_list=[]):
        super(ProjectBatchDataset, self).__init__()
        self.bert_input_list = bert_input_list
        self.bias_list = bias_list

    def __len__(self):
        assert len(self.bert_input_list) == len(self.bias_list), "Unmatched bert_input_list and bias_list"
        return len(self.bert_input_list)

    def empty(self):
        return self.__len__() == 0

    def reset(self):
        self.bert_input_list = []
        self.bias_list = []

    def __getitem__(self, index):
        return self.bert_input_list[index], self.bias_list[index]

    def __add__(self, bert_input, bias):
        self.bert_input_list.append(bert_input)
        self.bias_list.append(bias)

    def add(self, bert_input, bias):
        self.__add__(bert_input, bias)

