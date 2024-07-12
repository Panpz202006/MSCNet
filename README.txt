1.The introduction of models:
 (1)CNN_* are CNN-based models. Mamba_* are Mamba-based models.

2.Prepare datasets：
 (1)Push datastes to the data folder.
 (2)The datasets link are as follows:
      ISIC2017: https://challenge.isic-archive.com/data/#2017 
      ISIC2018: https://challenge.isic-archive.com/data/#2018
      PH2: https://www.dropbox.com/scl/fi/epzcoqeyr1v9qlv/PH2Dataset.rar?rlkey=6mt2jlvwfkditkyg12xdei6ux&e=1
 (3)Note: Alter your dataset path according to dataset/datasets.py.

3.Run codes:
 (1)Enter the main function, and then run it. 

4.The introduction of python files:
 (1)main.py: the entrance function of programs.
 (2)train.py: train model.
 (3)utils/lossfunction: get_metrics function is used for calculating metrics.
 (4)Note: If your data path is different with the operation of  dataset/dataset.py, you can alter the code relating dataset path in dataset/dataset.py.

