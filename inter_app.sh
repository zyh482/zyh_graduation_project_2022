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
SAVEDIR=checkpoints/iwed_${src}_${tgt}_${bedropout}_${mode}_h${hidden_dim}_gate110

python app/set_args.py $DATAPATH \
    --path $SAVEDIR/checkpoint_best.pt \
    --bert-model-name bert-base-uncased \
    --batch-size 1 --beam 5 --remove-bpe
CUDA_VISIBLE_DEVICES=3 uvicorn app.controller:app --reload | tee -a $SAVEDIR/app.log

