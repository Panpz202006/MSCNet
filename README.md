👋 [GitHub Repo stars: https://github.com/Panpz202006/MSCNet/tree/xyq_branch]

👋 [Abstract]

> In recent years, medical image segmentation has garnered widespread attention. However, existing methods typically face numerous challenges in addressing medical image segmentation: 1) blurred edges, 2) cluttered backgrounds, 3) complex structures, and 4) increased parameters and computational load. These problems result in the significant edge incompleteness and misrecognition of salient information. To address these challenges, in this paper, we propose a lightweight context extraction and edge enhancement network (LCENet) based on an Encoder-Decoder structure. Specifically, LCENet consists of three parts: an encoder, a Multi-scale Context(MSC) module, and Cross-level Fusion(CLF). The MSC module includes the Context Fusion Enhancement (CFE) module, the IG module, and residual connections. The encoder adopts the UltraLight VM-UNet backbone to generate six layers of feature embeddings with local details and long-range dependency relationships. Then, the MSC extracts eight different features through various receptive fields and edge enhancement to obtain multi-scale features containing fine-grained, local details, and global information. These features are then concatenated and fed into the IG module, which fuses the multi-scale features to obtain both local and global information and mitigate the impact of edge blurring. Subsequently, residual connections and activation functions are used to obtain features with strong anti-interference capabilities, containing detailed and salient information. Finally, the CLF module utilizes a cross-gating mechanism to leverage deep semantic information for salient guidance and shallow texture information for detail supplementation. Through element-wise addition, it fully utilizes the salient and detailed information to obtain more comprehensive features. Extensive experiments conducted on the ISIC2018, ISIC2017, and PH$^{2}$ datasets demonstrate that LCENet proves its superiority across eight evaluation metrics.

![network](https://github.com/user-attachments/assets/7a85f1e1-6474-48f3-8f2d-c92d6a84cfe0)

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

- **pre-training files**:

> the format of the file path: /checkpoints/backbone_name/dataset_name/filename.pth

  
## Run

- **example**:

> enter the directory where train.py is located. 

> train model, `python train.py --datasets ISIC2018 --backbone VGG`, which denotes to train on ISIC2018 datasets and to adopt VGG as backbone. Concrete information see train.py, please. Training records is saved to ./log folder, and pre-training file is saved to ./checkpoints/VGG.

> test model, `python train.py --datasets ISIC2018 --backbone UltraLight_VM_UNet`, which denotes to test on ISIC2018 datasets and to adopt UltraLight_VM_UNet as backbone. Concrete information see test.py, please. Testing records is saved to ./log folder, pre-training file is saved to ./checkpoints/UltraLight_VM_UNet, and testing results are saved to ./Test/UltraLight_VM_UNet/images.

- **prediction maps**:
  
> the format of the file path:  /Test/backbone_name/dataset_name/picture_name.png

  
## Comparison

- **quantitation**:
![image](https://github.com/user-attachments/assets/db408a6a-8ecf-4f7c-8a42-2f3f2f41ba29)




- **vision**:
<img width="1422" alt="comparative" src="https://github.com/user-attachments/assets/6ddae633-2daa-45f2-b661-76bbb280bf17">

