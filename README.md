👋 [GitHub Repo stars: https://github.com/Panpz202006/MSCNet/tree/xyq_branch]

\[ English | [中文](README_zh.md) \]

## Table of Contents

- [Preparation](#Preparation)
- [Run](#Run)
- [Comparison](#Comparison)


## Preparation

- **environments**: 
(1)configure
```bash
 pytorch2.2.2+cu118 
```
(2)install packages:
```bash
timm, mamba_ssm and tqdm packages.
```


- **datasets**: 

(1)download datasets:
```bash
ISIC2017(https://challenge.isic-archive.com/data/#2017)
ISIC2018(https://challenge.isic-archive.com/data/#2018)
PH2(https://www.dropbox.com/scl/fi/epzcoqeyr1v9qlv/PH2Dataset.rar?rlkey=6mt2jlvwfkditkyg12xdei6ux&e=1). 
```
(2)organize folders:
```bash
put ISIC2017 datasets into ./data/ISIC2017 folder, ISIC2018 datasets into ./data/ISIC2018 folder, and PH2 datasets into ./data/PH2 folder.
```

- **pre-training**: this part is not provided now.

## Run

- **example**:
(1)enter the directory where train.py is located. 
(2)for training model, run followed command:
```bash
python train.py --datasets ISIC2018 --backbone VGG
```
This command denotes to train on ISIC2018 datasets and to adopt VG as backbone. Concrete information see train.py, please. Training records is saved to ./log folder. 
(2)for testing model, run followed command:
```bash
python train.py --datasets ISIC2018 --backbone VGG
```
This command denotes to test on ISIC2018 datasets and to adopt VG as backbone. Concrete information see test.py, please. Testing records is saved to ./log folder.


## Comparison

- **quantitation**:No
- **vision**:No
