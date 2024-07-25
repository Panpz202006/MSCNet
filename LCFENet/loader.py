import sys
import os
from micro import TRAIN, VAL
sys.path.append(os.getcwd())
from datasets.dataset import ISIC2017_Datasets, ISIC2018_Datasets, PH2_Datasets
from utils.transforms import Test_Transformer, Train_Transformer
from torch.utils.data import DataLoader

def get_loader(datasets,batch_size,image_size,mode):
    if mode==TRAIN:
        transformer=Train_Transformer(image_size)
    else:
        transformer=Test_Transformer(image_size)
        batch_size=1
    if datasets=='ISIC2018':
        dataset=ISIC2018_Datasets(mode=mode,transformer=transformer)
    elif datasets=='ISIC2017':
        dataset=ISIC2017_Datasets(mode=mode,transformer=transformer)
    elif datasets=='PH2':
        dataset=PH2_Datasets(mode=mode,transformer=transformer)
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        pin_memory=True,
                        num_workers=2,
                        drop_last=True)
    return loader