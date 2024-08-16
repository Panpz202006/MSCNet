# LCENet

👋 [GitHub Repo stars: https://github.com/Panpz202006/MSCNet/tree/xyq_branch]

Abstract

In recent years, medical image segmentation has garnered significant attention. However, existing methods still struggle to effectively address issues such as blurred edges, background interferences, and complex structures.To address these issues, we propose a lightweight context extraction and edge enhancement network (LCENet), which consists of two key modules: the Multi-Scale Context Module (MSC) and the Cross-Layer Fusion Module (CLF). Specifically, MSC extracts multi-scale context information through parallel dilated convolutions to establish long-range dependency, aiming to obtain multi-scale semantic information. It also accurately delineates the boundaries of lesion areas from this information based on both spatial and channel dimensions. The CLF leverages a cross-gated mechanism to extract shallow texture details to assist high-level semantic information in locating salient regions, and employs an attention mechanism to enrich high-level semantic features, thereby efficiently highlighting the lesion areas.Extensive experiments on the ISIC2018, ISIC2017, and PH$^{2}$ datasets show that LCENet excels in five evaluation metrics.

![network](https://github.com/user-attachments/assets/deeb9bdd-903c-410b-9e6b-84cbc549c848)



## Table of Contents

- [Main Environments](#Main Environments)
- [Datasets](#Datasets)
- [Train the LCENet](#Train the LCENet)
- [Test the LCENet](#Test the LCENet)
- [Comparison With State of the Arts](#Comparison With State of the Arts)
- [Acknowledgement](#Acknowledgement)


## Main Environments

The environment installation procedure can be followed by UltraLight-VM-UNet, or by following the steps below (python=3.8):

```
conda create -n LCENet python=3.8
conda activate LCENet
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
pip install packaging
pip install timm==0.4.12
pip install pytest chardet yacs termcolor
pip install submitit tensorboardX
pip install triton==2.0.0
pip install causal_conv1d==1.0.0  # causal_conv1d-1.0.0+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install mamba_ssm==1.0.1  # mmamba_ssm-1.0.1+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install scikit-learn matplotlib thop h5py SimpleITK scikit-image medpy yacs
```

## Datasets: 

- Download datasets: ISIC2017 [https://challenge.isic-archive.com/data/#2017](https://challenge.isic-archive.com/data/#2017), ISIC2018 [https://challenge.isic-archive.com/data/#2018](https://challenge.isic-archive.com/data/#2018), and PH2 [https://www.dropbox.com/scl/fi/epzcoqeyr1v9qlv/PH2Dataset.rar?rlkey=6mt2jlvwfkditkyg12xdei6ux&e=1](https://www.dropbox.com/scl/fi/epzcoqeyr1v9qlv/PH2Dataset.rar?rlkey=6mt2jlvwfkditkyg12xdei6ux&e=1).

- Folder organization: put ISIC2017 datasets into ./data/ISIC2017 folder, ISIC2018 datasets into ./data/ISIC2018 folder, and PH2 datasets into ./data/PH2 folder.
  
## Training

```
python train.py --datasets ISIC2018 --backbone UltraLight_VM_UNet
training records is saved to ./log folder
pre-training file is saved to ./checkpoints/UltraLight_VM_UNet
concrete information see ./LCENet/train.py, please
```

## Testing

```
python train.py --datasets ISIC2018 --backbone UltraLight_VM_UNet
testing records is saved to ./log
training file is saved to ./checkpoints/UltraLight_VM_UNet
testing results are saved to ./Test/UltraLight_VM_UNet/images
concrete information see ./LCENet/test.py, please
```
  
## Comparison With State of the Arts

![image](https://github.com/user-attachments/assets/db408a6a-8ecf-4f7c-8a42-2f3f2f41ba29)

<img width="1422" alt="comparative" src="https://github.com/user-attachments/assets/6ddae633-2daa-45f2-b661-76bbb280bf17">

##Acknowledgments

Thanks to UltraLight-VM-UNet for his outstanding works.
