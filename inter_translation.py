# -*- coding:utf-8 -*-
# CREATED BY: zhangyuhan
# CREATED ON: 2022/4/17 4:01 PM
# LAST MODIFIED ON:
# AIM:
import torch
import click
from loguru import logger
from bert import BertTokenizer
from fairseq import utils, checkpoint_utils, tasks, options


parser = options.get_generation_parser()
args = options.parse_args_and_arch(parser)
use_cuda = torch.cuda.is_available() and not args.cpu

task = tasks.setup_task(args)
# Set dictionaries
try:
    src_dict = getattr(task, 'source_dictionary', None)
except NotImplementedError:
    src_dict = None
tgt_dict = task.target_dictionary
bert_tokenizer = BertTokenizer.from_pretrained(args.bert_model_name)
# Load alignment dictionary for unknown word replacement
# (None if no unknown word replacement, empty if no path to align dictionary)
align_dict = utils.load_align_dict(args.replace_unk)

# Load ensemble
print(f'| loading model(s) from {args.path}')
models, _model_args = checkpoint_utils.load_model_ensemble(
    args.path.split(':'),
    arg_overrides=eval(args.model_overrides),
    task=task,
    bert_ratio=args.bert_ratio if args.change_ratio else None,
    encoder_ratio=args.encoder_ratio if args.change_ratio else None,
    geargs=args,
)

# Optimize ensemble for generation
for model in models:
    model.make_generation_fast_(
        beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
        need_attn=args.print_alignment,
    )
    if args.fp16:
        model.half()
    if use_cuda:
        model.cuda()

generator = task.build_generator(args)


def preprocess(src_sentence: str):
    src_sentence = src_sentence.lower()
    print(src_sentence)
    src_tokens = src_dict.encode_line(src_sentence)
    src_bert_tokens = bert_tokenizer.tokenize(src_sentence)
    src_bert_tokens = bert_tokenizer.convert_tokens_to_ids(src_bert_tokens)
    sample = {
        'net_input': {
            'src_tokens': src_tokens.unsqueeze(0),
            'src_lengths': torch.tensor(len(src_tokens), dtype=int).unsqueeze(0),
            'bert_input': torch.tensor(src_bert_tokens, dtype=int).unsqueeze(0),
        }
    }
    sample = utils.move_to_cuda(sample) if use_cuda else sample
    return sample


# @click.command()
# @click.option('--sentence', prompt='sentence', help='English sentence to translate')
def translation(sentence):
    sample = preprocess(sentence)
    hypo = task.inference_step(generator, models, sample, prefix_tokens=None)
    hypo = hypo[0][0]

    src_tokens = utils.strip_pad(sample['net_input']['src_tokens'], tgt_dict.pad())
    hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
        hypo_tokens=hypo['tokens'].int().cpu(),
        src_str=src_dict.string(src_tokens, args.remove_bpe),
        alignment=hypo['alignment'].int().cpu() if hypo['alignment'] is not None else None,
        align_dict=align_dict,
        tgt_dict=tgt_dict,
        remove_bpe=args.remove_bpe,
    )
    print(hypo_str)


if __name__ == '__main__':
    while True:
        sentence = input('sentence: ')
        try:
            translation(sentence)
        except Exception as E:
            print(E.__str__())

