#!/usr/bin/env bash
nvidia-smi

cd /data/zhangyuhan/project
python3 -c "import torch; print(torch.__version__)"

src=en
tgt=de
bedropout=0.3
mode='sample'
hidden_dim=1000
bias_dim=512
ARCH=transformer_s3_iwslt_de_en
DATAPATH=data-bin/iwslt14.tokenized.en-de
SAVEDIR=checkpoints/iwed_${src}_${tgt}_${bedropout}_${mode}

CUDA_VISIBLE_DEVICES=6 python generate.py $DATAPATH \
       --path $SAVEDIR/checkpoint_nmt.pt \
       --bert-model-name bert-base-uncased \
       --model-overrides "{'arch': '${ARCH}', 'hidden_dim': ${hidden_dim}, 'bias_dim': ${bias_dim}, 'residual': False,
       'project_path': '${SAVEDIR}/project_model_h${hidden_dim}.best'}" \
       --batch-size 128 --beam 5 --remove-bpe | tee -a $SAVEDIR/testing_project_h${hidden_dim}.log

# | Translated 6750 sentences (1356750 tokens) in 231.7s (29.13 sentences/s, 5854.97 tokens/s)
# | Generate test with beam=5: BLEU4 = 0.00, 0.0/0.0/0.0/0.0 (BP=1.000, ratio=8.577, syslen=1078564, reflen=125755)