# zyh_graduation_project_2022

## 环境配置
```
Python version >= 3.5
PyTorch version == 1.0.0/1.1.0

pip install --editable .
```

## 目录结构



## 快速启动
- ### 下载数据集
    ```
  cd examples/translation
  bash prepare-iwslt14.sh
  ```
- ### 数据预处理
    运行makedataforbert.sh前注意修改路径
    ```
  bash makedataforbert.sh en|de
  cd ../..
  TEXT=examples/translation/iwslt14.tokenized.de-en
  python preprocess.py --source-lang en --target-lang de \
  --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
  --destdir data-bin/iwslt14.tokenized.en-de --joined-dictionary \
  --bert-model-name bert-base-uncased
  ```
- ### 训练模型
  运行前注意修改路径
    ```
  bash entrypoint.sh
  ```
- ### 模型评估
  运行前注意修改路径
    ```
  bash evaluate.sh
  ```
