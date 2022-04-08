# -*- coding:utf-8 -*-
# CREATED BY: zhangyuhan
# CREATED ON: 2022/4/7 8:19 PM
# LAST MODIFIED ON:
# AIM:
"""Model Structure for VAE Project"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils


class VAEProjectModel(nn.Module):
    def __init__(self, args):
        super(VAEProjectModel, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(args.bert_out_dim, args.hidden_dim)
        self.fc21 = nn.Linear(args.hidden_dim, args.bias_dim)
        self.fc22 = nn.Linear(args.hidden_dim, args.bias_dim)
        self.dropout = args.dropout
        self.activation_fn = utils.get_activation_fn(activation=getattr(args, 'activation_fn', 'relu'))
        self.activation_dropout = getattr(args, 'activation_dropout', 0)

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """
        project bert-layer-output into sentence-embedding
        params:
            x: bert-output [batch_size, sequence_length, bert_out_dim]
        return:
            [batch_size, bias_dim]
        """
        x = x[:, 0, :]      # cls
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        mu = self.fc21(x)
        logvar = self.fc22(x)
        x = self.reparametrize(mu, logvar)
        # x = F.dropout(x, p=self.dropout, training=self.training)
        return x, mu, logvar
