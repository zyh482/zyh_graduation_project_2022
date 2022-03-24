#!/usr/bin/env bash
lng=$1
echo "src lng $lng"
for sub  in train valid test
do
    sub=/data/zhangyuhan/project/examples/translation/iwslt14.tokenized.de-en/${sub}
    sed -r 's/(@@ )|(@@ ?$)//g' ${sub}.${lng} > ${sub}.bert.${lng}.tok
    # ../mosesdecoder/scripts/tokenizer/detokenizer.perl -l $lng < ${sub}.bert.${lng}.tok > ${sub}.bert.${lng}
    /data/zhangyuhan/project/examples/translation/mosesdecoder/scripts/tokenizer/detokenizer.perl -l $lng < ${sub}.bert.${lng}.tok > ${sub}.bert.${lng}
    rm ${sub}.bert.${lng}.tok
done