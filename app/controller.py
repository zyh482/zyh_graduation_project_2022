# -*- coding:utf-8 -*-
# CREATED BY: zhangyuhan
# CREATED ON: 2022/4/16 10:55 PM
# LAST MODIFIED ON:
# AIM:
import re
import time
import fastapi
import torch
from fastapi import FastAPI
from loguru import logger
from bert import BertTokenizer
from app.schema import Request, Response
from fairseq import options, utils, tasks, checkpoint_utils, bleu


def register_echo(app):
    '''register app echo apis 注册心跳和根目录

    Args:
        app: FastAPI Instance

    Returns:
        app: registered FastAPI Instance
    '''

    @app.get('/health')
    def healty():
        return 'ok'

    @app.get('/')
    def hello():
        return f'Hello FastAPI {time.ctime()}'

    @app.get('/err/{sleep}')
    def sleep(sleep):
        logger.info(sleep)
        time.sleep(int(sleep))
        logger.info(f"{sleep} ok...")
        return sleep

    return app


def server(task, args):
    # Load ensemble
    logger.info(f'| loading model(s) from {args.path}')
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

    # Generate and compute BLEU score
    if args.sacrebleu:
        scorer = bleu.SacrebleuScorer()
    else:
        scorer = bleu.Scorer(tgt_dict.pad(), tgt_dict.eos(), tgt_dict.unk())

    return models, generator, scorer

def preprocess(src_sentence: str, src_dict, bert_tokenizer):
    print(src_sentence)
    src_tokens = src_dict.encode_line(src_sentence, append_eos=args.append_eos)
    src_bert_tokens = bert_tokenizer.encode(src_sentence)
    sample = {
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': len(src_sentence),
            'bert_input': src_bert_tokens,
        }
    }
    print(sample)
    sample = utils.move_to_cuda(sample) if use_cuda else sample
    return sample



app = fastapi.FastAPI()
app = register_echo(app)

args = torch.load('app/args.config')
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

models, generator, scorer = server(task, args)
logger.info('Successfully load models and generator.')


def on_success(code=0, data={}, message="success!"):
    return {'code': code, 'data': data, 'message': message}


def on_failed(code=1, data={}, message="failed!"):
    return {'code': code, 'data': data, 'message': message}


@app.post('/translation')
async def translation(request: Request):
    sentence = request.sentence
    sample = preprocess(sentence, src_dict, bert_tokenizer)
    hypo = task.inference_step(generator, models, sample, prefix_tokens=0)

    src_tokens = utils.strip_pad(sample['net_input']['src_tokens'], tgt_dict.pad())
    hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
        hypo_tokens=hypo['tokens'].int().cpu(),
        src_str=src_dict.string(src_tokens, args.remove_bpe),
        alignment=hypo['alignment'].int().cpu() if hypo['alignment'] is not None else None,
        align_dict=align_dict,
        tgt_dict=tgt_dict,
        remove_bpe=args.remove_bpe,
    )

    response = Response(sentence=hypo_str)
    return on_success(data=response.dict(exclude_none=True))

