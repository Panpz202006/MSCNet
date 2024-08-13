👋 [GitHub Repo stars: https://github.com/Panpz202006/MSCNet/tree/xyq_branch]

👋 [Abstract]

> In recent years, medical image segmentation has garnered significant attention. However, existing methods still struggle to effectively address issues such as blurred edges, background interferences, and complex structures.To address these issues, we propose a lightweight context extraction and edge enhancement network (LCENet), which consists of two key modules: the Multi-Scale Context Module (MSC) and the Cross-Layer Fusion Module (CLF). Specifically, MSC extracts multi-scale context information through parallel dilated convolutions to establish long-range dependency, aiming to obtain multi-scale semantic information. It also accurately delineates the boundaries of lesion areas from this information based on both spatial and channel dimensions. The CLF leverages a cross-gated mechanism to extract shallow texture details to assist high-level semantic information in locating salient regions, and employs an attention mechanism to enrich high-level semantic features, thereby efficiently highlighting the lesion areas.Extensive experiments on the ISIC2018, ISIC2017, and PH$^{2}$ datasets show that LCENet excels in five evaluation metrics.

![connection](https://github.com/user-attachments/assets/6fbb695b-ef82-4d59-898c-6e328e93039a)

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

