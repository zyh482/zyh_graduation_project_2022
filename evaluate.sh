#!/usr/bin/env bash
nvidia-smi

cd /data/zhangyuhan/bert-nmt
python3 -c "import torch; print(torch.__version__)"

src=en
tgt=de
bedropout=0.3
ARCH=transformer_s2_iwslt_de_en
DATAPATH=data-bin/iwslt14.tokenized.en-de
SAVEDIR=checkpoints/iwed_${src}_${tgt}_${bedropout}

CUDA_VISIBLE_DEVICES=6 python generate.py $DATAPATH \
    --path $SAVEDIR/checkpoint_best.pt \
    --bert-model-name bert-base-uncased \
    --batch-size 128 --beam 5 --remove-bpe | tee -a $SAVEDIR/testing.log

#| Translated 6750 sentences (156854 tokens) in 28.5s (236.92 sentences/s, 5505.55 tokens/s)
#| Generate test with beam=5: BLEU4 = 30.16, 63.8/36.6/23.4/15.5 (BP=0.993, ratio=0.993, syslen=124892, reflen=125755)