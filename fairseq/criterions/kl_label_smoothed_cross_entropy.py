# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
import torch

from fairseq import utils

from . import FairseqCriterion, register_criterion
EVAL_BLEU_ORDER = 4


@register_criterion('kl_label_smoothed_cross_entropy')
class KLdLabelSmoothedCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        # fmt: on

    def forward(self, model, sample, bias=None, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output, mu, logvar = model(**sample['net_input'], bias=bias)
        loss, nll_loss = self.compute_loss(model, net_output, mu, logvar, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, mu, logvar, sample, reduce=True):
        sen_kld = -0.5*torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        target = model.get_targets(sample, net_output)
        sen_len = torch.sum(target.ne(self.padding_idx), dim=-1)
        nll_kld = sen_kld/sen_len
        nll_kld = nll_kld.view(1, -1).repeat([1, target.shape[-1]])

        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = target.view(-1, 1)
        nll_kld = nll_kld.view(-1, 1)
        non_pad_mask = target.ne(self.padding_idx)

        nll_loss = -lprobs.gather(dim=-1, index=target)[non_pad_mask]
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)[non_pad_mask]
        nll_loss = nll_loss + nll_kld[non_pad_mask]
        if reduce:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()
        eps_i = self.eps / lprobs.size(-1)
        loss = (1. - self.eps) * nll_loss + eps_i * smooth_loss
        return loss, nll_loss

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        def sum_logs(key):
            import torch

            result = sum(log.get(key, 0) for log in logging_outputs)
            if torch.is_tensor(result):
                result = result.cpu()
            return result

        def compute_bleu():
            try:
                from sacrebleu.metrics import BLEU
                comp_bleu = BLEU.compute_bleu
            except ImportError:
                # compatibility API for sacrebleu 1.x
                import sacrebleu
                comp_bleu = sacrebleu.compute_bleu

            counts, totals = [], []
            for i in range(EVAL_BLEU_ORDER):
                counts.append(sum_logs("_bleu_counts_" + str(i)))
                totals.append(sum_logs("_bleu_totals_" + str(i)))
            bleu = comp_bleu(
                correct=counts,
                total=totals,
                sys_len=sum_logs('_bleu_sys_len'),
                ref_len=sum_logs('_bleu_ref_len'),
                smooth_method='exp',
            )
            return round(bleu.score, 2)

        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2),
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
            'bleu_score': compute_bleu(),
        }
