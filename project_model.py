# -*- coding:utf-8 -*-
# CREATED BY: zhangyuhan
# CREATED ON: 2022/3/28 11:25 PM
# LAST MODIFIED ON:
# AIM:
"""Model Structure for project"""

import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils


class ProjectModel(nn.Module):
    """FFN"""
    def __init__(self, args):
        super(ProjectModel, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(args.bert_out_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.bias_dim)
        self.dropout = args.dropout
        self.activation_fn = utils.get_activation_fn(activation=getattr(args, 'activation_fn', 'relu'))
        self.activation_dropout = getattr(args, 'activation_dropout', 0)

    def forward(self, x):
        """
        project bert-layer-output into sentence-embedding
        params:
            x: bert-output [batch_size, sequence_length, bert_out_dim]
        return:
            [batch_size, bias_dim]
        """
        x = x[:, 0, :]      # cls
        if self.args.residual:
            residual = xgit
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        if self.args.residual:
            x = residual + x
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x

