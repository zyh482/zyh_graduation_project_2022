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
mkdir -p $SAVEDIR
if [ ! -f $SAVEDIR/checkpoint_nmt.pt ]
then
    cp checkpoints/iwed_en_de_0.3/checkpoint_best.pt $SAVEDIR/checkpoint_nmt.pt
    # cp /data/zhangyuhan/fairseq/checkpoints/iwslt14_en_de/checkpoint_best.pt $SAVEDIR/checkpoint_nmt.pt
fi
if [ ! -f "$SAVEDIR/checkpoint_last.pt" ]
then
warmup="--warmup-from-nmt --reset-lr-scheduler --reset-optimizer --reset-dataloader"
else
warmup=""
fi

CUDA_VISIBLE_DEVICES=5 nohup python train.py $DATAPATH \
-a $ARCH --optimizer adam --lr 0.0005 -s $src -t $tgt --label-smoothing 0.1 \
--dropout 0.3 --max-tokens 4000 --min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
--criterion kl_label_smoothed_cross_entropy --max-update 300000 \
--warmup-updates 4000 --warmup-init-lr '1e-07' --fp16 \
--adam-betas '(0.9,0.98)' --save-dir $SAVEDIR --share-all-embeddings $warmup \
--encoder-bert-dropout --encoder-bert-dropout-ratio $bedropout \
--hidden-dim $hidden_dim --bias-dim 512 \
--eval-bleu \
--beam 5 --max-len-a 1.2 --max-len-b 10 \
--eval-bleu-remove-bpe \
--eval-bleu-print-samples | tee -a $SAVEDIR/training.log > $SAVEDIR/train.output &
