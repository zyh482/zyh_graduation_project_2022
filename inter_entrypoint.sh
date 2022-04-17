#!/usr/bin/env bash
nvidia-smi

cd /data/zhangyuhan/project
python3 -c "import torch; print(torch.__version__)"

src=en
tgt=de
bedropout=0.3
#ARCH=transformer_s2_iwslt_de_en
#DATAPATH=data-bin/iwslt14.tokenized.en-de
#SAVEDIR=checkpoints/iwed_${src}_${tgt}_${bedropout}
mode='vae'
hidden_dim=640
ARCH=transformer_s3_iwslt_de_en
DATAPATH=data-bin/iwslt14.tokenized.en-de
SAVEDIR=checkpoints/iwed_${src}_${tgt}_${bedropout}_${mode}_h${hidden_dim}_gate110


CUDA_VISIBLE_DEVICES=6 python inter_translation.py $DATAPATH \
    --path $SAVEDIR/checkpoint_best.pt \
    --bert-model-name bert-base-uncased \
    --batch-size 1 --beam 5 --remove-bpe | tee -a $SAVEDIR/interaction.log
