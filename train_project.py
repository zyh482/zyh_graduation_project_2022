# -*- coding:utf-8 -*-
# CREATED BY: zhangyuhan
# CREATED ON: 2022/3/28 4:00 PM
# LAST MODIFIED ON:
# AIM:
"""
Train a project model to project bert-encode-output of source-sentence into recover-bias of target-sentence
"""

import collections
import math
import os.path

import torch
import argparse

from fairseq import checkpoint_utils, distributed_utils, options, progress_bar, tasks, utils
from fairseq.data import iterators
from fairseq.meters import AverageMeter, StopwatchMeter
from project_model import ProjectModel
from project_trainer import ProjectTrainer


def train_project(args):
    print(args)
    # Initialize model
    model = ProjectModel(args)
    print(model)
    print('| num. model params: {} (num. trained: {})'.format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    ))
    # Build trainer
    trainer = ProjectTrainer(args, model)
    # load dataloader
    train_dataloader = trainer.get_split_iterator('train')
    valid_dataloader = trainer.get_split_iterator('valid')

    max_epoch = args.max_epoch or math.inf
    max_update = args.max_update or math.inf
    lr = trainer.get_lr()
    epoch = 0
    best_valid_loss = math.inf

    path_prefix = 'project_model'
    if args.residual:
        path_prefix = path_prefix+'_residual'
    path_prefix += f'_h{args.hidden_dim}'

    train_meter = StopwatchMeter()
    train_meter.start()
    # Train until the learning rate gets too small
    while lr > args.min_lr and epoch < max_epoch and trainer.get_num_updates() < max_update:
        # train for one epoch
        train(args, trainer, train_dataloader, epoch)
        # validate
        if not args.disable_validation and epoch % args.validate_interval == 0:
            validate(args, trainer, valid_dataloader, epoch)
            valid_avg_loss = trainer.get_meter('valid_loss').avg
            # save the best model
            if valid_avg_loss < best_valid_loss:
                best_valid_loss = valid_avg_loss
                trainer.save_model(os.path.join(args.save_dir, f'{path_prefix}.best'))
        else:
            valid_avg_loss = None

        # use validation loss to update the learning rate
        lr = trainer.lr_step(epoch, valid_avg_loss)
        # save the latest model
        trainer.save_model(os.path.join(args.save_dir, f'{path_prefix}.latest'))

        epoch += 1
        # reset training meters
        for k in trainer.meters.keys():
            meter = trainer.get_meter(k)
            if meter is not None:
                meter.reset()

    train_meter.stop()
    print('| done training in {:.1f} seconds'.format(train_meter.sum))


def train(args, trainer, dataloader, epoch):
    """Train the model for one epoch"""
    progress = progress_bar.build_progress_bar(args, dataloader, epoch=epoch, no_progress_bar='simple')
    for i, batch in enumerate(dataloader):
        train_loss = trainer.train_step(batch)
        train_loss = train_loss.item()
        # log mid-epoch stats
        stats = get_training_stats(trainer)
        for k, meter in stats.items():
            if isinstance(meter, AverageMeter):
                stats[k] = meter.avg
        progress.log(stats, tag='train', step=stats['num_updates'])

        if stats['num_updates'] >= args.max_update:
            break
    # log end-of-epoch stats
    stats = get_training_stats(trainer)
    for k, meter in stats.items():
        if isinstance(meter, AverageMeter):
            stats[k] = stats[k].avg
    progress.print(stats, tag='train', step=stats['num_updates'])


def get_training_stats(trainer):
    stats = collections.OrderedDict()
    stats['loss'] = trainer.get_meter('train_loss')
    stats['bsz'] = trainer.get_meter('bsz')
    stats['num_updates'] = trainer.get_num_updates()
    stats['lr'] = trainer.get_lr()
    stats['gnorm'] = trainer.get_meter('gnorm')
    stats['clip'] = trainer.get_meter('clip')
    stats['train_wall'] = trainer.get_meter('train_wall')
    return stats


def validate(args, trainer, dataloader, epoch):
    """Evaluate the model on the validation set"""
    progress = progress_bar.build_progress_bar(args, dataloader, prefix='valid', epoch=epoch, no_progress_bar='simple')
    trainer.get_meter('valid_loss').reset()

    for i, batch in enumerate(dataloader):
        valid_loss = trainer.valid_step(batch)

    # log validation stats
    stats = get_valid_stats(trainer)
    for k, meter in stats.items():
        if isinstance(meter, AverageMeter):
            stats[k] = meter.avg
    progress.print(stats, tag='valid', step=stats['num_updates'])


def get_valid_stats(trainer):
    stats = collections.OrderedDict()
    stats['loss'] = trainer.get_meter('valid_loss')
    stats['bsz'] = trainer.get_meter('bsz')
    stats['num_updates'] = trainer.get_num_updates()
    return stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parser for project-model training')
    parser.add_argument('--data-dir', default=None, help='dir for loading dataset')
    parser.add_argument('--save-dir', default=None, help='dir for saving model.pt')
    parser.add_argument('--shuffle', action='store_true', help='shuffle for dataloader')
    parser.add_argument('--bert-model-name', default='bert-base-uncased', type=str)
    parser.add_argument('--bert-out-dim', default=768, type=int, help='dim of bert-output')
    parser.add_argument('--bert-output-layer', default=-1, type=int, help='output layer of bert')
    parser.add_argument('--bias-dim', default=512, type=int, help='dim of bias')
    parser.add_argument('--hidden-dim', default=768, type=int, help='dim of hidden states')
    parser.add_argument('--dropout', default=0.3, help='dropout ratio')
    parser.add_argument('--activation-fn', default='relu', help='activation function')
    parser.add_argument('--activation-dropout', default=0, help='dropout ratio for activation')
    parser.add_argument('--residual', action='store_true', help='residual net')
    parser.add_argument('--criterion', default='cosine', help='criterion, e.g., CosineEmbeddingLoss')
    parser.add_argument('--cosine-target', default=1, choices=[-1, 1], help='target for compute CosineEmbeddingLoss')
    parser.add_argument('--optimizer', default='adam', help='optimizer for gradient descent')
    parser.add_argument('--adam-betas', default='(0.9, 0.999)', metavar='B', help='betas for Adam optimizer')
    parser.add_argument('--adam-eps', type=float, default=1e-8, metavar='D', help='epsilon for Adam optimizer')
    parser.add_argument('--weight-decay', '--wd', default=0.0, type=float, metavar='WD', help='weight decay')
    parser.add_argument('--lr', default=[0.01], help='initial learning rate')
    parser.add_argument('--lr-scheduler', default='reduce_lr_on_plateau')
    parser.add_argument('--lr-patience', default=0, type=int,
                        help='Number of epochs with no improvement after which learning rate will be reduced.')
    parser.add_argument('--lr-shrink-factor', default=0.1, type=float, metavar='LS',
                        help='shrink factor for annealing, lr_new = (lr * lr_shrink_factor)')
    parser.add_argument('--lr-threshold', default=1e-9, type=float, metavar='LT',
                        help='Threshold for measuring the new optimum, to only focus on significant changes')
    parser.add_argument('--clip-norm', default=5, type=float, metavar='NORM', help='clip threshold of gradients')
    parser.add_argument('--max-epoch', default=None)
    parser.add_argument('--max-update', default=50000)
    parser.add_argument('--min-lr', default=1e-5, type=float)
    parser.add_argument('--disable-validation', action='store_true')
    parser.add_argument('--validate-interval', default=1)
    parser.add_argument('--no-progress-bar', action='store_true', help='disable progress bar')
    parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                        help='log progress every N batches (when progress bar is disabled)')
    parser.add_argument('--log-format', choices=['json', 'none', 'simple', 'tqdm'], default='simple', help='log format to use')
    parser.add_argument('--tensorboard-logdir', metavar='DIR', default='',
                        help='path to save logs for tensorboard, should match --logdir of running tensorboard')
    parser.add_argument("--tbmf-wrapper", action="store_true", help="[FB only] ")
    parser.add_argument('--seed', default=1, type=int, metavar='N', help='pseudo random number generator seed')
    args = parser.parse_args()

    train_project(args)
