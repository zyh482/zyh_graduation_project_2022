#!/usr/bin/env bash
nvidia-smi

cd /data/zhangyuhan/project
python3 -c "import torch; print(torch.__version__)"

src=en
tgt=de
bedropout=0.3
DATAPATH=data-bin/iwslt14.tokenized.en-de/${tgt}_bias
SAVEDIR=checkpoints/iwed_${src}_${tgt}_${bedropout}_sample


CUDA_VISIBLE_DEVICES=7 nohup python -u train_project.py \
--data-dir $DATAPATH --save-dir $SAVEDIR \
--bert-model-name 'bert-base-uncased' \
--lr-scheduler reduce_lr_on_plateau --lr-patience 3 --lr-shrink-factor 0.3 \
--residual \
--min-lr '1e-06' | tee -a $SAVEDIR/project_residual_training.log > $SAVEDIR/project_residual_train.out &

#--lr-scheduler reduce_lr_on_plateau --lr-patience 3 --lr-shrink-factor 0.8 \
