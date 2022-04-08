#!/usr/bin/env bash
nvidia-smi

cd /data/zhangyuhan/project
python3 -c "import torch; print(torch.__version__)"

src=en
tgt=de
bedropout=0.3
mode='vae'
hidden_dim=640
ARCH=transformer_s3_iwslt_de_en
DATAPATH=data-bin/iwslt14.tokenized.en-de
SAVEDIR=checkpoints/iwed_${src}_${tgt}_${bedropout}_${mode}_h${hidden_dim}

CUDA_VISIBLE_DEVICES=0 python generate.py $DATAPATH \
    --path $SAVEDIR/checkpoint_best.pt \
    --bert-model-name bert-base-uncased \
    --batch-size 128 --beam 5 --remove-bpe | tee -a $SAVEDIR/testing.log

#| Translated 6750 sentences (156075 tokens) in 28.3s (238.65 sentences/s, 5518.22 tokens/s)
#| Generate test with beam=5: BLEU4 = 30.18, 64.0/36.9/23.6/15.6 (BP=0.989, ratio=0.989, syslen=124354, reflen=125755)