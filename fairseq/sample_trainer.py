# -*- coding:utf-8 -*-
# CREATED BY: zhangyuhan
# CREATED ON: 2022/3/28 4:25 PM
# LAST MODIFIED ON:
# AIM:
"""
Train corresponding bias for sample
"""
import collections
import csv
from collections import OrderedDict
from itertools import chain
import math
import os
import sys

import torch

from fairseq import checkpoint_utils, distributed_utils, models, optim, utils, progress_bar
from fairseq.meters import AverageMeter, StopwatchMeter, TimeMeter
from fairseq.optim import lr_scheduler


class SampleTrainer(object):
    def __init__(self, args, task, model, criterion, dummy_batch=None, oom_batch=None):
        self.args = args
        self.task = task
        # copy model and criterion to current device
        self.criterion = criterion
        self._model = model
        if self.args.eval_bleu:
            self.sequence_generator = self.task.build_generator(self.args)
        self.cuda = torch.cuda.is_available() and not args.cpu
        if args.fp16:
            self._model = self._model.half()
        if self.cuda:
            self.criterion = self.criterion.cuda()
            self._model = self._model.cuda()

        self._dummy_batch = dummy_batch
        self._oom_batch = oom_batch or dummy_batch

        self._lr_scheduler = None
        self._num_updates = 0
        self._optim_history = None
        self._optimizer = None
        self._prev_grad_norm = None
        self._wrapped_model = None

        self.init_meters(args)

    def init_meters(self, args):
        self.meters = OrderedDict()
        self.meters['train_loss'] = AverageMeter()
        self.meters['train_nll_loss'] = AverageMeter()
        self.meters['bleu_score'] = AverageMeter()
        self.meters['wps'] = TimeMeter()       # words per second
        self.meters['ups'] = TimeMeter()       # updates per second
        self.meters['wpb'] = AverageMeter()    # words per batch
        self.meters['bsz'] = AverageMeter()    # sentences per batch
        self.meters['gnorm'] = AverageMeter()  # gradient norm
        self.meters['clip'] = AverageMeter()   # % of updates clipped
        self.meters['oom'] = AverageMeter()    # out of memory
        if args.fp16:
            self.meters['loss_scale'] = AverageMeter()  # dynamic loss scale
        self.meters['wall'] = TimeMeter()      # wall time in seconds
        self.meters['train_wall'] = StopwatchMeter()  # train wall time in seconds

    @property
    def model(self):
        if self._wrapped_model is None:
            if self.args.distributed_world_size > 1:
                self._wrapped_model = models.DistributedFairseqModel(
                    self.args, self._model,
                )
            else:
                self._wrapped_model = self._model
        return self._wrapped_model

    @property
    def optimizer(self):
        if self._optimizer is None:
            self._build_optimizer()
        return self._optimizer

    @property
    def lr_scheduler(self):
        if self._lr_scheduler is None:
            self._build_optimizer()  # this will initialize self._lr_scheduler
        return self._lr_scheduler

    def _build_optimizer(self, params=None):
        params = list(filter(lambda p: p.requires_grad, params))
        if self.args.fp16:
            if self.cuda and torch.cuda.get_device_capability(0)[0] < 7:
                print('| WARNING: your device does NOT support faster training with --fp16, '
                      'please switch to FP32 which is likely to be faster')
            if self.args.memory_efficient_fp16:
                self._optimizer = optim.MemoryEfficientFP16Optimizer.build_optimizer(self.args, params)
            else:
                self._optimizer = optim.FP16Optimizer.build_optimizer(self.args, params)
        else:
            if self.cuda and torch.cuda.get_device_capability(0)[0] >= 7:
                print('| NOTICE: your device may support faster training with --fp16')
            self._optimizer = optim.build_optimizer(self.args, params)

        # We should initialize the learning rate scheduler immediately after
        # building the optimizer, so that the initial learning rate is set.
        self._lr_scheduler = lr_scheduler.build_lr_scheduler(self.args, self.optimizer)
        self._lr_scheduler.step_update(0)

    def load_checkpoint(
            self,
            filename,
            reset_optimizer=True,
            reset_lr_scheduler=True,
            optimizer_overrides=None,
            reset_meters=True,
            warmup_from_nmt=False,
    ):
        """Load all training state from a checkpoint file."""
        extra_state, self._optim_history, last_optim_state = None, [], None

        if os.path.exists(filename):
            state = checkpoint_utils.load_checkpoint_to_cpu(filename)

            # load model parameters
            try:
                # loaded_results = self.get_model().load_state_dict(state['model'], strict=False if warmup_from_nmt else True)
                self.get_model().load_state_dict(state['model'], strict=False if warmup_from_nmt else True)
            except Exception:
                raise Exception(
                    'Cannot load model parameters from checkpoint, '
                    'please ensure that the architectures match.'
                )

            # if warmup_from_nmt:
            #     assert len(loaded_results.unexpected_keys) == 0
            #     for missing_key in loaded_results.missing_keys:
            #         assert 'bert' in missing_key, 'key {} is missing'.format(missing_key)
            extra_state = state['extra_state']
            self._optim_history = state['optimizer_history']
            last_optim_state = state['last_optimizer_state']

        assert reset_optimizer and reset_lr_scheduler, 'if set --train-sample-bias, ' \
                                                       'please set --reset-optimizer --reset-lr-scheduler'

        if extra_state is not None:
            epoch = extra_state['train_iterator']['epoch']
            print('| loaded checkpoint {} (epoch {} @ {} updates)'.format(
                filename, epoch, self.get_num_updates()))

            if 'train_meters' in extra_state:
                self.meters.update(extra_state['train_meters'])
                del extra_state['train_meters']

                # reset TimeMeters, since their start times don't make sense anymore
                for meter in self.meters.values():
                    if isinstance(meter, TimeMeter):
                        meter.reset()
        else:
            print('| no existing checkpoint found {}'.format(filename))

        return extra_state

    def get_train_iterator(self, epoch, combine=True):
        print('| loading train data for sample training')
        self.task.load_dataset(self.args.train_subset, epoch=epoch, combine=combine)
        return self.task.get_batch_iterator(
            dataset=self.task.dataset(self.args.train_subset),
            max_tokens=self.args.max_tokens,
            max_sentences=self.args.max_sentences,
            max_positions=utils.resolve_max_positions(
                self.task.max_positions(),
                self.model.max_positions(),
                ),
            ignore_invalid_inputs=True,
            required_batch_size_multiple=self.args.required_batch_size_multiple,
            seed=self.args.seed,
            num_shards=self.args.distributed_world_size,
            shard_id=self.args.distributed_rank,
            num_workers=self.args.num_workers,
            epoch=epoch,
        )

    def get_split_iterator(self, split, epoch, combine=True):
        """ Load a given dataset split.
        Args:
            split (str): name of split (e.g., train, valid, test)
        """
        print(f'| loading {split} data for sample training')

        self.task.load_dataset(split, epoch=epoch, combine=combine)
        return self.task.get_batch_iterator(
            dataset=self.task.dataset(split),
            max_tokens=self.args.max_tokens,
            max_sentences=self.args.max_sentences,
            max_positions=utils.resolve_max_positions(
                self.task.max_positions(),
                self.model.max_positions(),
                ),
            ignore_invalid_inputs=True,
            required_batch_size_multiple=self.args.required_batch_size_multiple,
            seed=self.args.seed,
            num_shards=self.args.distributed_world_size,
            shard_id=self.args.distributed_rank,
            num_workers=self.args.num_workers,
            epoch=epoch,
        )

    def train_sample(self, sample, path, dummy_batch=False, raise_oom=False):
        """Do gradient descent for samples to find best bias"""
        if self._dummy_batch is None:
            self._dummy_batch = sample

        self._set_seed()
        self.model.eval()
        self.criterion.eval()

        # random initialize bias
        # bias = torch.randn([sample['nsentences'], self.args.decoder_embed_dim])
        # Xavier initialize bias
        bias = torch.empty([sample['nsentences'], self.args.decoder_embed_dim])
        torch.nn.init.xavier_uniform_(bias, gain=torch.nn.init.calculate_gain('relu'))
        # move to the same device
        sample, bias = self._prepare_sample(sample, bias)

        if sample is None:
            return
        bias.requires_grad = True
        bias = torch.nn.parameter.Parameter(bias)
        # show sample size
        print(f"| {sample['nsentences']} sentences {sample['net_input']['src_tokens'].shape} "
              f"with targets {sample['target'].shape}")

        # reset optimizer, learning-rate and num-updates
        self._build_optimizer([bias])
        self.zero_grad()
        lr = self.get_lr()
        self.set_num_updates(0)
        self.meters['train_wall'].start()
        # initialize criterion
        train_epoch = 0
        train_nll_loss = math.inf
        best_bleu, best_bias = 0, None
        max_update = self.args.max_update or math.inf
        min_nll_loss = min(self.args.min_loss_scale, 2/sample['target'].shape[-1])
        min_lr = self.args.min_lr or 1e-5
        ooms = 0
        progress = progress_bar.build_progress_bar(self.args, [sample], no_progress_bar='simple')
        # Train bias until the nll_loss gets too small or learning rate gets too small
        # or num_updates gets too big or bleu score gets to threshold
        while train_nll_loss > min_nll_loss and lr > min_lr \
                and self.get_num_updates() < max_update and best_bleu < self.args.bleu_threshold:
            try:
                # forward
                loss, sample_size, logging_output = self.criterion(self.model, sample, bias)
                self._optimizer.backward(loss)
                # compute bleu
                if self.args.eval_bleu:
                    bleu = self.task.inference_with_bleu(self.sequence_generator, sample, self.model, bias)
                    logging_output['bleu_score'] = round(bleu.score, 2)

                # clip loss
                # logging_output = self.task.aggregate_logging_outputs(
                #     logging_outputs, self.criterion
                # )
                # sample_size = self.task.grad_denom(sample_sizes, self.criterion)
                logging_output['loss'] = logging_output['loss'] / logging_output['sample_size'] / math.log(2)
                logging_output['nll_loss'] = logging_output['nll_loss'] / logging_output['ntokens'] / math.log(2)

                # backward
                # save previous bias before descent
                pre_bias = bias
                # normalize grads by sample size
                sample_size = self.task.grad_denom([sample_size], self.criterion)
                self._optimizer.multiply_grads(self.args.distributed_world_size/float(sample_size))
                # clip grads
                grad_norm = self.optimizer.clip_grad_norm(self.args.clip_norm)
                self._prev_grad_norm = grad_norm
                # take an optimization step
                self.optimizer.step()
                self.set_num_updates(self.get_num_updates() + 1)
                self.lr_step(train_epoch)

                # update meters
                ntokens = logging_output.get('ntokens', 0)
                nsentences = logging_output.get('nsentences', 0)
                self.meters['gnorm'].update(grad_norm)
                self.meters['clip'].update(1. if grad_norm > self.args.clip_norm and self.args.clip_norm > 0 else 0.)
                self.meters['wps'].update(ntokens)
                self.meters['ups'].update(1.)
                self.meters['wpb'].update(ntokens)
                self.meters['bsz'].update(nsentences)
                self.meters['train_loss'].update(logging_output.get('loss', 0), sample_size)
                if 'train_acc' in self.meters:
                    self.meters['train_acc'].update(
                        logging_output.get('acc', 0), sample_size)
                if 'nll_loss' in logging_output:
                    self.meters['train_nll_loss'].update(logging_output.get('nll_loss', 0), ntokens)
                self.meters['bleu_score'].update(logging_output.get('bleu_score', 0))
                if logging_output.get('bleu_score', 0) > best_bleu:
                    best_bleu = logging_output.get('bleu_score', 0)
                    best_bias = pre_bias

                train_nll_loss = self.meters['train_nll_loss'].avg
            except RuntimeError as e:
                raise e

            # log stats
            if train_epoch % self.args.log_epoch_interval == 0:
                stats = self.get_train_stats()
                for k, meter in stats.items():
                    if isinstance(meter, AverageMeter):
                        stats[k] = meter.avg
                progress.print(stats, tag='train', step=self.get_num_updates())

            train_epoch += 1

        stats = self.get_train_stats()
        for k, meter in stats.items():
            if isinstance(meter, AverageMeter):
                stats[k] = meter.avg
        progress.print(stats, tag='train', step=self.get_num_updates())

        self.meters['train_wall'].stop()
        print('| done training in {:.1f} seconds with best_bleu {:.2f}'.format(self.meters['train_wall'].sum, best_bleu))
        torch.save({'sample': sample, 'bias': best_bias, 'bleu': best_bleu}, path)
        print(f'| save sample-bias into {path}')
        # reset training meters
        for key in self.meters.keys():
            meter = self.get_meter(key)
            if meter:
                meter.reset()

    def get_train_stats(self):
        stats = collections.OrderedDict()
        stats['loss'] = self.get_meter('train_loss')
        if self.get_meter('train_nll_loss').count > 0:
            nll_loss = self.get_meter('train_nll_loss')
            stats['nll_loss'] = nll_loss
        else:
            nll_loss = self.get_meter('train_loss')
        stats['ppl'] = utils.get_perplexity(nll_loss.avg)
        stats['num_updates'] = self.get_num_updates()
        stats['lr'] = self.get_lr()
        stats['bleu_score'] = self.get_meter('bleu_score')
        return stats

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

    def _prepare_sample(self, sample, bias):
        if sample is None or len(sample) == 0:
            return None
        if self.cuda:
            sample = utils.move_to_cuda(sample)
            bias = utils.move_to_cuda(bias)
        return sample, bias

    def _set_seed(self):
        # Set seed based on args.seed and the update number so that we get
        # reproducible results when resuming from checkpoints
        seed = self.args.seed + self.get_num_updates()
        torch.manual_seed(seed)
        if self.cuda:
            torch.cuda.manual_seed(seed)
