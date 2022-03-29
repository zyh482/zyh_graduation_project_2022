# -*- coding:utf-8 -*-
# CREATED BY: zhangyuhan
# CREATED ON: 2022/3/28 4:23 PM
# LAST MODIFIED ON:
# AIM:
"""
Train a project model
"""
import re
from collections import OrderedDict
from itertools import chain
import math
import os
import sys

import torch
import torch.utils.data
from bert.modeling import BertModel
from bert.tokenization import BertTokenizer
from fairseq import optim
from fairseq.meters import AverageMeter, StopwatchMeter, TimeMeter
from fairseq.optim import lr_scheduler
from fairseq.progress_bar import progress_bar
from project_dataset import ProjectBatchDataset


class ProjectTrainer(object):
    def __init__(self, args, model):
        self.args = args
        self._model = model
        self._bert_model = BertModel.from_pretrained(args.bert_model_name)
        self._bert_tokenizer = BertTokenizer.from_pretrained(args.bert_model_name)

        assert self.args.bert_out_dim == self._bert_model.hidden_size
        self.pad = self._bert_tokenizer.pad()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._model = self._model.to(self.device)
        self._bert_model.to(self.device)

        self._lr_scheduler = None
        self._num_updates = 0
        self._optim_history = None
        self._optimizer = None
        self._prev_grad_norm = None
        self._criterion = None

        self.init_meters()
        self.init_split_paths()

    def init_meters(self):
        self.meters = OrderedDict()
        self.meters['train_loss'] = AverageMeter()
        self.meters['valid_loss'] = AverageMeter()
        self.meters['bsz'] = AverageMeter()     # sentence per batch
        self.meters['gnorm'] = AverageMeter()   # gradient norm
        self.meters['clip'] = AverageMeter()    # % of updates clipped
        self.meters['wall'] = TimeMeter()      # wall time in seconds
        self.meters['train_wall'] = StopwatchMeter()  # train wall time in seconds

    def init_split_paths(self):
        self.split_paths = {'train': [], 'valid': [], 'test': []}
        assert os.path.exists(self.args.data_dir), f"Not found dir {self.args.data_dir}"
        # traverse all files in data_dir
        g = os.walk(self.args.data_dir)
        for path, dir_list, file_list in g:
            for file in file_list:
                if re.search('train', file):
                    self.split_paths['train'].append(os.path.join(path, file))
                if re.search('valid', file):
                    self.split_paths['valid'].append(os.path.join(path, file))
                if re.search('test', file):
                    self.split_paths['test'].append(os.path.join(path, file))
        print(f"| Find {len(self.split_paths['train'])} train files, {len(self.split_paths['valid'])} valid files "
              f"and {len(self.split_paths['test'])} test files")

    @property
    def model(self):
        return self._model

    @property
    def optimizer(self):
        if self._optimizer is None:
            self._build_optimizer()
        return self._optimizer

    @property
    def lr_scheduler(self):
        if self._lr_scheduler is None:
            self._build_optimizer()
        return self._lr_scheduler

    @property
    def criterion(self):
        if self._criterion is None:
            self._build_criterion()
        return self._criterion

    def _build_optimizer(self):
        params = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        self._optimizer = optim.build_optimizer(self.args, params)
        # Initialize the learning rate scheduler
        self._lr_scheduler = lr_scheduler.build_lr_scheduler(self.args, self.optimizer)
        self._lr_scheduler.step_update(0)

    def _build_criterion(self):
        if self.args.criterion == 'cosine':
            self._criterion = torch.nn.CosineEmbeddingLoss(reduction='mean')

    def save_model(self, filename):
        torch.save(self.model.state_dict(), filename)

    def load_model(self, filename):
        assert os.path.exists(filename), f'| no existing model found {filename}'
        self._model.load_state_dict(torch.load(filename))

    def load_dataset(self, split):
        """
        Load a given dataset split.
        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        # bert_input_list, bias_list = [], []
        dataset = ProjectBatchDataset()
        if not dataset.empty():
            dataset.reset()
        count = 0
        for file_path in self.split_paths[split]:
            try:
                data = torch.load(file_path)
                bert_input = data['sample']['net_input']['bert_input']
                bias = data['bias']
                assert bert_input.shape[0] == bias.shape[0] and self.args.bias_dim == bias.shape[-1]

                # bert_input_list = bert_input_list + [t.tolist() for t in bert_input]
                # bias_list = bias_list + [t.tolist() for t in bias]
                dataset.add(bert_input, bias)
                count += bias.shape[0]
                # for b_input, bia in zip(bert_input.split(batch_size), bias.split(batch_size)):
                #     print(b_input.shape, bia.shape)
                #     bert_padding_mask = b_input.eq(self.pad)
                #     bert_output, _ = self._bert_model(b_input,
                #                                       attention_mask=~bert_padding_mask,
                #                                       output_all_encoded_layers=True)
                #     bert_output = bert_output[self.args.bert_output_layer]
                #     print(bert_output.shape, bia.shape)
                #     bert_output_list = bert_output_list + [t for t in bert_output]
                #     bias_list = bias_list + [t for t in bia]
            except Exception as e:
                print(e)
                print(f"| Failed to load {split} data from {file_path}")

        print(f"| Successfully load {split} data from {len(dataset)} files, {count} samples")
        return dataset

    def get_split_iterator(self, split):
        dataset = self.load_dataset(split)
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
        return torch.utils.data.DataLoader(dataset,
                                           shuffle=self.args.shuffle,
                                           sampler=sampler,
                                           )

    def bert_preprocess(self, bert_input):
        """
        input: bert-input [batch_size, sequence_length]
        return: bert-output [batch_size, sequence_length, bert-out-dim]
        """
        bert_padding_mask = bert_input.eq(self.pad)
        bert_output, _ = self._bert_model(bert_input,
                                          attention_mask=~bert_padding_mask,
                                          output_all_encoded_layers=True)
        bert_output = bert_output[self.args.bert_output_layer]
        bert_output.to(self.device)
        return bert_output

    def train_step(self, batch):
        """Do forward, backward and update parameters"""
        torch.manual_seed(self.args.seed+self.get_num_updates())
        self.model.train()
        self.zero_grad()

        self.meters['train_wall'].start()

        bert_input, bias = tuple(t.to(self.device) for t in batch)
        bert_input = bert_input.squeeze()
        bias = bias.squeeze()
        bert_output = self.bert_preprocess(bert_input)
        output = self.model(bert_output)

        bsz = bias.shape[0]
        target = self.args.cosine_target * torch.ones(bsz, dtype=int).to(self.device)
        loss = self.criterion(output, bias, target)
        loss.backward()
        # clip grads
        grad_norm = self.optimizer.clip_grad_norm(self.args.clip_norm)
        self.optimizer.step()
        self.set_num_updates(self.get_num_updates()+1)
        # update meters
        self.meters['train_loss'].update(loss.item(), bsz)
        self.meters['bsz'].update(bsz)
        self.meters['gnorm'].update(grad_norm.item())
        self.meters['clip'].update(1. if grad_norm > self.args.clip_norm > 0 else 0.)
        self.meters['train_wall'].stop()

        return loss

    def valid_step(self, batch):
        with torch.no_grad():
            self.model.eval()
            bert_input, bias = tuple(t.to(self.device) for t in batch)
            bert_input = bert_input.squeeze()
            bias = bias.squeeze()
            bert_output = self.bert_preprocess(bert_input)
            output = self.model(bert_output)

            bsz = bias.shape[0]
            target = self.args.cosine_target * torch.ones(bsz, dtype=int).to(self.device)
            loss = self.criterion(output, bias, target)
        # update meters
        self.meters['valid_loss'].update(loss.item(), bsz)
        self.meters['bsz'].update(bsz)
        self.meters['train_wall'].stop()

        return loss

    def zero_grad(self):
        self.optimizer.zero_grad()

    def lr_step(self, epoch, val_loss=None):
        """Adjust the learning rate based on the validation loss."""
        _lr = self.lr_scheduler.step(epoch, val_loss)
        # prefer updating the LR based on the number of steps
        return self.lr_step_update()

    def lr_step_update(self):
        """Update the learning rate after each update."""
        return self.lr_scheduler.step_update(self.get_num_updates())

    def get_lr(self):
        """Get the current learning rate."""
        return self.optimizer.get_lr()

    def get_model(self):
        """Get the (non-wrapped) model instance."""
        return self._model

    def get_meter(self, name):
        """Get a specific meter by name."""
        if name not in self.meters:
            return None
        return self.meters[name]

    def get_num_updates(self):
        """Get the number of parameters updates."""
        return self._num_updates

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        self._num_updates = num_updates
        self.lr_step_update()


