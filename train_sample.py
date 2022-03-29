# -*- coding:utf-8 -*-
# CREATED BY: zhangyuhan
# CREATED ON: 2022/3/28 3:59 PM
# LAST MODIFIED ON:
# AIM:
"""
Train bias for each sample to recover target-sentence
"""
import math

import torch
from torch.nn.parameter import Parameter
from torch.optim import Adam
from fairseq import checkpoint_utils, distributed_utils, options, progress_bar, tasks, utils
from fairseq.data import iterators
from fairseq.sample_trainer import SampleTrainer
from fairseq.meters import AverageMeter, StopwatchMeter


def train_sample(args):
    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(args)

    # Build model
    model = task.build_model(args)
    # Fix model parameters
    for param in model.parameters():
        param.requires_grad = False
    # Build criterion
    criterion = task.build_criterion(args)
    # print model
    print(model)
    print('| model {}, criterion {}'.format(args.arch, criterion.__class__.__name__))
    print('| num. model params: {} (num. trained: {})'.format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    ))

    # Build trainer
    trainer = SampleTrainer(args, task, model, criterion)
    print('| training on {} GPUs'.format(args.distributed_world_size))
    print('| max tokens per GPU = {} and max sentences per GPU = {}'.format(
        args.max_tokens,
        args.max_sentences,
    ))

    # load the pretrained model from checkpoint
    # --reset-optimizer --reset-lr-schedule --reset-dataloader
    extra_state, epoch_itr = checkpoint_utils.load_checkpoint(args, trainer)

    # load samples from split_subset
    assert args.split in ['train', 'valid', 'test'], f"Invalid split: {args.split}"
    if args.split != 'train':
        epoch_itr = trainer.get_split_iterator(args.split, epoch=0)

    train_meter = StopwatchMeter()
    train_meter.start()
    # Initialize data iterator
    # Update parameters every N batches
    update_freq = args.update_freq[epoch_itr.epoch - 1] \
        if epoch_itr.epoch <= len(args.update_freq) else args.update_freq[-1]
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=args.fix_batches_to_gpus,
        shuffle=(epoch_itr.epoch >= args.curriculum),
    )
    itr = iterators.GroupedIterator(itr, update_freq)
    progress = progress_bar.build_progress_bar(args, itr, epoch_itr.epoch, no_progress_bar='simple')

    for i, samples in enumerate(progress, start=epoch_itr.iterations_in_epoch):
        print(f'sample {i}')
        trainer.train_sample(samples[0], task.bias_save_path+f'.{i}')

    train_meter.stop()
    print('| Done training in {:.1f} seconds'.format(train_meter.sum))
