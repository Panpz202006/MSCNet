import random
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
import torch

class Datasets(Dataset):
    def __init__(self,datasets_path,transformer):
        super(Datasets,self).__init__()
        images_path=os.path.join(datasets_path,'images')
        gts_path=os.path.join(datasets_path,'gt')
        images_list=sorted(os.listdir(images_path))
        gts_list=sorted(os.listdir(gts_path))
        self.data=[]

        for i in range(len(images_list)):
            image_path=images_path+'/'+images_list[i]
            mask_path=gts_path+'/'+gts_list[i]
            self.data.append([image_path, mask_path])
        self.transformer=transformer
    
    def __getitem__(self, index):
        image_path, gt_path=self.data[index]
        image = Image.open(image_path).convert('RGB')
        image=np.array(image)
        image = np.transpose(image, axes=(2, 0, 1))
        gt = Image.open(gt_path).convert('L')
        gt = np.array(gt)
        gt=np.expand_dims(gt, axis=2) / 255
        gt = np.transpose(gt, axes=(2, 0, 1))
        image, gt = self.transformer((image, gt))
        return image,gt

    def __len__(self):
        return len(self.data)

class PH2_Datasets(Dataset):
    def __init__(self,datasets_path,transformer,val=False):
        super().__init__()
        self.data=[]
        datasets_list=sorted(os.listdir(datasets_path))
        random.shuffle(datasets_list)

        for path in datasets_list:
            image_path=datasets_path+'/'+path+'/'+path+'_Dermoscopic_Image/'+path+'.bmp'
            gt_path=datasets_path+'/'+path+'/'+path+'_lesion/'+path+'_lesion.bmp'
            self.data.append([image_path, gt_path])
        print(len(self.data))
        limit=int(len(self.data)*0.8)
        if val:
            self.data=self.data[limit:]
        else:
            self.data=self.data[:limit]
        print(len(self.data))
        self.transformer=transformer
    
    def __getitem__(self, index):
        image_path, gt_path=self.data[index]
        image = Image.open(image_path).convert('RGB')
        image=np.array(image)
        image = np.transpose(image, axes=(2, 0, 1))
        gt = Image.open(gt_path).convert('L')
        gt = np.array(gt)
        gt=np.expand_dims(gt, axis=2) / 255
        gt = np.transpose(gt, axes=(2, 0, 1))
        image, gt = self.transformer((image, gt))
        return image,gt

    def __len__(self):
        return len(self.data)
