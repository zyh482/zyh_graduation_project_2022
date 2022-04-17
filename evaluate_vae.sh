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

CUDA_VISIBLE_DEVICES=3 python generate.py $DATAPATH \
    --path $SAVEDIR/checkpoint_best.pt \
    --bert-model-name bert-base-uncased \
    --batch-size 128 --beam 5 --remove-bpe | tee -a $SAVEDIR/testing.log

#| Translated 6750 sentences (156075 tokens) in 28.3s (238.65 sentences/s, 5518.22 tokens/s)
#| Generate test with beam=5: BLEU4 = 30.18, 64.0/36.9/23.6/15.6 (BP=0.989, ratio=0.989, syslen=124354, reflen=125755)

# gate010
#| Translated 6750 sentences (157501 tokens) in 29.7s (227.20 sentences/s, 5301.32 tokens/s)
#| Generate test with beam=5: BLEU4 = 30.63, 64.0/37.1/23.9/15.8 (BP=0.995, ratio=0.995, syslen=125139, reflen=125755)

# gate100
#| Translated 6750 sentences (159484 tokens) in 28.8s (233.97 sentences/s, 5528.09 tokens/s)
#| Generate test with beam=5: BLEU4 = 30.27, 63.4/36.5/23.4/15.5 (BP=1.000, ratio=1.008, syslen=126761, reflen=125755)