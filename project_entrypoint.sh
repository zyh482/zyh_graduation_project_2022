#!/usr/bin/env bash
nvidia-smi

cd /data/zhangyuhan/project
python3 -c "import torch; print(torch.__version__)"

src=en
tgt=de
bedropout=0.3
hidden_dim=1000
DATAPATH=data-bin/iwslt14.tokenized.en-de/${tgt}_bias
SAVEDIR=checkpoints/iwed_${src}_${tgt}_${bedropout}_sample


CUDA_VISIBLE_DEVICES=7 nohup python -u train_project.py \
--data-dir $DATAPATH --save-dir $SAVEDIR \
--bert-model-name 'bert-base-uncased' \
--hidden-dim $hidden_dim \
--lr-scheduler reduce_lr_on_plateau --lr-patience 3 --lr-shrink-factor 0.3 \
--min-lr '1e-06' | tee -a $SAVEDIR/project_h${hidden_dim}_training.log > $SAVEDIR/project_h${hidden_dim}_train.out &

#--lr-scheduler reduce_lr_on_plateau --lr-patience 3 --lr-shrink-factor 0.8 \
