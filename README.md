# LCENet

👋 [GitHub Repo stars: https://github.com/Panpz202006/MSCNet/tree/xyq_branch]

💥 [Overview] 💥

In recent years, medical image segmentation has garnered significant attention. However, existing methods still struggle to effectively address issues such as blurred edges, background interferences, and complex structures.To address these issues, we propose a lightweight context extraction and edge enhancement network (LCENet), which consists of two key modules: the Multi-Scale Context Module (MSC) and the Cross-Layer Fusion Module (CLF). Specifically, MSC extracts multi-scale context information through parallel dilated convolutions to establish long-range dependency, aiming to obtain multi-scale semantic information. It also accurately delineates the boundaries of lesion areas from this information based on both spatial and channel dimensions. The CLF leverages a cross-gated mechanism to extract shallow texture details to assist high-level semantic information in locating salient regions, and employs an attention mechanism to enrich high-level semantic features, thereby efficiently highlighting the lesion areas.Extensive experiments on the ISIC2018, ISIC2017, and PH$^{2}$ datasets show that LCENet excels in five evaluation metrics.

![network](https://github.com/user-attachments/assets/deeb9bdd-903c-410b-9e6b-84cbc549c848)



## Table of Contents

- [Preparation](#Preparation)
- [Training](#Training)
- [Results](#Results)


## Preparation

Requirements: 

- python 3.10
- numpy 1.26.3
- pytorch 2.2.2
- torchvision 0.17.2
- timm 0.4.12
- mamba_ssm 1.2.0

Datasets: 

- Download datasets: ISIC2017 [https://challenge.isic-archive.com/data/#2017](https://challenge.isic-archive.com/data/#2017), ISIC2018 [https://challenge.isic-archive.com/data/#2018](https://challenge.isic-archive.com/data/#2018), and PH2 [https://www.dropbox.com/scl/fi/epzcoqeyr1v9qlv/PH2Dataset.rar?rlkey=6mt2jlvwfkditkyg12xdei6ux&e=1](https://www.dropbox.com/scl/fi/epzcoqeyr1v9qlv/PH2Dataset.rar?rlkey=6mt2jlvwfkditkyg12xdei6ux&e=1).

- Folder organization: put ISIC2017 datasets into ./data/ISIC2017 folder, ISIC2018 datasets into ./data/ISIC2018 folder, and PH2 datasets into ./data/PH2 folder.
  
## Training

Enter the directory: ./LCENet/ 

Training

```
python train.py --datasets ISIC2018 --backbone UltraLight_VM_UNet
training records is saved to ./log folder
pre-training file is saved to ./checkpoints/VGG.
Concrete information see ./LCENet/train.py, please. 
```

Evaluation:

```
python train.py --datasets ISIC2018 --backbone UltraLight_VM_UNet
testing records is saved to ./log folder
training file is saved to ./checkpoints/UltraLight_VM_UNet
testing results are saved to ./Test/UltraLight_VM_UNet/images.
Concrete information see ./LCENet/test.py, please. 
```
  
## Results

- **quantitation**:
![image](https://github.com/user-attachments/assets/db408a6a-8ecf-4f7c-8a42-2f3f2f41ba29)


- **vision**:
<img width="1422" alt="comparative" src="https://github.com/user-attachments/assets/6ddae633-2daa-45f2-b661-76bbb280bf17">
