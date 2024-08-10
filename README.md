👋 [GitHub Repo stars: https://github.com/Panpz202006/MSCNet/tree/xyq_branch]

## Table of Contents

- [Preparation](#Preparation)
- [Run](#Run)
- [Comparison](#Comparison)


## Preparation

- **environments**: 

> configure `pytorch2.2.2+cu118` 

> install `timm, mamba_ssm and tqdm` packages.


- **datasets**: 

> download datasets: ISIC2017`https://challenge.isic-archive.com/data/#2017`, ISIC2018`https://challenge.isic-archive.com/data/#2018`, and PH2`https://www.dropbox.com/scl/fi/epzcoqeyr1v9qlv/PH2Dataset.rar?rlkey=6mt2jlvwfkditkyg12xdei6ux&e=1`.  

> foolder organization: put ISIC2017 datasets into ./data/ISIC2017 folder, ISIC2018 datasets into ./data/ISIC2018 folder, and PH2 datasets into ./data/PH2 folder.


## Run

- **example**:

> enter the directory where train.py is located. 

> train model, `python train.py --datasets ISIC2018 --backbone VGG`, which denotes to train on ISIC2018 datasets and to adopt VGG as backbone. Concrete information see train.py, please. Training records is saved to ./log folder, and pre-training file is saved to ./checkpoints/VGG.

> test model, `python train.py --datasets ISIC2018 --backbone UltraLight_VM_UNet`, which denotes to test on ISIC2018 datasets and to adopt UltraLight_VM_UNet as backbone. Concrete information see test.py, please. Testing records is saved to ./log folder, pre-training file is saved to ./checkpoints/UltraLight_VM_UNet, and testing results are saved to ./Test/UltraLight_VM_UNet/images.

## Comparison

- **quantitation**:No
- **vision**:
<img width="1422" alt="comparative" src="https://github.com/user-attachments/assets/6ddae633-2daa-45f2-b661-76bbb280bf17">

