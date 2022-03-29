#!/usr/bin/env bash
nvidia-smi

cd /data/zhangyuhan/project
python3 -c "import torch; print(torch.__version__)"

src=en
tgt=de
bedropout=0.3
mode='sample'
split='test'
ARCH=transformer_s2_iwslt_de_en
DATAPATH=data-bin/iwslt14.tokenized.en-de
SAVEDIR=checkpoints/iwed_${src}_${tgt}_${bedropout}_${mode}
mkdir -p $SAVEDIR
if [ ! -f $SAVEDIR/checkpoint_nmt.pt ]
then
    cp checkpoints/iwed_en_de_0.3/checkpoint_best.pt $SAVEDIR/checkpoint_nmt.pt
    # cp /data/zhangyuhan/fairseq/checkpoints/iwslt14_en_de/checkpoint_best.pt $SAVEDIR/checkpoint_nmt.pt
fi
if [ ! -f "$SAVEDIR/checkpoint_last.pt" ]
then
warmup="--warmup-from-nmt --reset-lr-scheduler"
else
warmup=""
fi

CUDA_VISIBLE_DEVICES=5 nohup python train.py $DATAPATH \
-a $ARCH --optimizer adam --lr 0.03 -s $src -t $tgt --label-smoothing 0.1 \
--dropout 0.3 --max-tokens 4000 --min-lr '1e-05' --weight-decay 0.0001 \
--criterion label_smoothed_cross_entropy --max-update 1000 \
--train-mode $mode --split $split \
--log-epoch-interval 25 --sample-savedir $DATAPATH/${tgt}_bias --bleu-threshold 100 \
--reset-optimizer --reset-lr-scheduler --reset-dataloader --reset-meters \
--min-loss-scale 0.1 \
--lr-scheduler reduce_lr_on_plateau --lr-patience 3 --lr-shrink-factor 0.8 \
--adam-betas '(0.9,0.98)' --save-dir $SAVEDIR --share-all-embeddings $warmup \
--encoder-bert-dropout --encoder-bert-dropout-ratio $bedropout \
--eval-bleu \
--beam 5 --max-len-a 1.2 --max-len-b 10 \
--eval-bleu-remove-bpe | tee -a $SAVEDIR/${split}_training.log > $SAVEDIR/${split}_train.output &
# --eval-bleu-print-samples
# --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr '0.01' \