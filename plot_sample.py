import argparse
import os
import re

import torch
import matplotlib.pyplot as plt


def plot_sample(args):
    print(args)
    len_list = []
    bleu_list = []
    g = os.walk(args.dir)
    for path, dir_list, file_list in g:
        for file in file_list:
            if re.search(args.split, file):
                dict = torch.load(os.path.join(path, file))
                print(dict)
                tgt_tokens = dict['sample']['target']
                bleu = dict['bleu']
                if tgt_tokens.shape[-1] <= 150:
                    len_list.append(tgt_tokens.shape[-1])
                    bleu_list.append(bleu)
                assert len(len_list) == len(bleu_list)
                if len(len_list) >= 200:
                    break
        if len(len_list) >= 200:
            break

    plt.scatter(len_list, bleu_list)
    plt.xlabel("length")
    plt.ylabel("BLEU")
    plt.savefig(os.path.join(args.dir, f'len-bleu_{args.split[:2]}.png'))
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("phraser for plot")
    parser.add_argument('--split', choices=['train', 'valid', 'test'], default='test')
    parser.add_argument('--dir', default='data-bin/iwslt14.tokenized.en-de/de_bias')
    args = parser.parse_args()
    plot_sample(args)
