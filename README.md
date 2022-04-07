# zyh_graduation_project_2022

## 环境配置
```
Python version >= 3.5
PyTorch version == 1.0.0/1.1.0

pip install --editable .
```

## 目录结构



## 快速启动
**运行脚本前注意检查路径**
- ### 下载数据集
    ```
  cd examples/translation
  bash prepare-iwslt14.sh
  ```
- ### 数据预处理
  ```
  bash makedataforbert.sh en|de
  cd ../..
  TEXT=examples/translation/iwslt14.tokenized.de-en
  python preprocess.py --source-lang en --target-lang de \
  --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
  --destdir data-bin/iwslt14.tokenized.en-de --joined-dictionary \
  --bert-model-name bert-base-uncased
  ```
- ### 训练bert-nmt
  ```
  bash model-entrypoint.sh
  ```
- ### 模型评估
  ```
  bash evaluate.sh
  ```
- ### 梯度下降求sentence-recover-bias
  ```
  bash sample-entrypoint.sh
  ```
- ### 训练project-model
  ```
  bash project-entrypoint.sh
  ```
- ### 评估加入project后的模型
  ```
  bash evaluate_project.sh
  ```


## 参考文献
```
@inproceedings{
Zhu2020Incorporating,
title={Incorporating BERT into Neural Machine Translation},
author={Jinhua Zhu and Yingce Xia and Lijun Wu and Di He and Tao Qin and Wengang Zhou and Houqiang Li and Tieyan Liu},
booktitle={International Conference on Learning Representations},
year={2020},
url={https://openreview.net/forum?id=Hyl7ygStwB}
}
```