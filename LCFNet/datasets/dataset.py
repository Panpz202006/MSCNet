import random
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
import torch

from micro import TEST, TRAIN, VAL


class ISIC2018_Datasets(Dataset):
    def __init__(self,mode,transformer):
        super().__init__()
        cwd=os.getcwd()
        ''' 按照Inter-Scale Dependency Modeling for Skin Lesion Segmentation with Transformer-based Network, 划分数据集
        '''
        # if mode==TRAIN:
        #     gts_path=os.path.join(cwd,'data','ISIC2018','ISIC2018_Task1_Training_GroundTruth','ISIC2018_Task1_Training_GroundTruth')
        #     images_path=os.path.join(cwd,'data','ISIC2018','ISIC2018_Task1-2_Training_Input','ISIC2018_Task1-2_Training_Input')
        # elif mode==VAL:
        #     gts_path=os.path.join(cwd,'data','ISIC2018','ISIC2018_Task1_Validation_GroundTruth','ISIC2018_Task1_Validation_GroundTruth')
        #     images_path=os.path.join(cwd,'data','ISIC2018','ISIC2018_Task1-2_Validation_Input','ISIC2018_Task1-2_Validation_Input')
        # elif mode==TEST:
        #     gts_path=os.path.join(cwd,'data','ISIC2018','ISIC2018_Task1_Test_GroundTruth','ISIC2018_Task1_Test_GroundTruth')
        #     images_path=os.path.join(cwd,'data','ISIC2018','ISIC2018_Task1-2_Test_Input','ISIC2018_Task1-2_Test_Input')
        
        gts_path=os.path.join(cwd,'data','ISIC2018','ISIC2018_Task1_Training_GroundTruth','ISIC2018_Task1_Training_GroundTruth')
        images_path=os.path.join(cwd,'data','ISIC2018','ISIC2018_Task1-2_Training_Input','ISIC2018_Task1-2_Training_Input')
        
        images_list=sorted(os.listdir(images_path))
        images_list = [item for item in images_list if "jpg" in item]
        gts_list=sorted(os.listdir(gts_path))
        gts_list = [item for item in gts_list if "png" in item]
        self.data=[]
        for i in range(len(images_list)):
            image_path=images_path+'/'+images_list[i]
            mask_path=gts_path+'/'+gts_list[i]
            self.data.append([image_path, mask_path])
        self.transformer=transformer
        random.shuffle(self.data)
        if mode==TRAIN:
            self.data=self.data[:1815]
        elif mode==VAL:
            self.data=self.data[1815:2074]
        elif mode==TEST:
            self.data=self.data[2074:2594]
        print(len(self.data))

    


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



class ISIC2017_Datasets(Dataset):
    def __init__(self,mode,transformer):
        super().__init__()
        cwd=os.getcwd()
        if mode==TRAIN:
            gts_path=os.path.join(cwd,'data','ISIC2017','ISIC-2017_Training_Part1_GroundTruth','ISIC-2017_Training_Part1_GroundTruth')
            images_path=os.path.join(cwd,'data','ISIC2017','ISIC-2017_Training_Data','ISIC-2017_Training_Data')
        elif mode==VAL:
            gts_path=os.path.join(cwd,'data','ISIC2017','ISIC-2017_Validation_Part1_GroundTruth','ISIC-2017_Validation_Part1_GroundTruth')
            images_path=os.path.join(cwd,'data','ISIC2017','ISIC-2017_Validation_Data','ISIC-2017_Validation_Data')
        elif mode==TEST:
            gts_path=os.path.join(cwd,'data','ISIC2017','ISIC-2017_Test_v2_Part1_GroundTruth','ISIC-2017_Test_v2_Part1_GroundTruth')
            images_path=os.path.join(cwd,'data','ISIC2017','ISIC-2017_Test_v2_Data','ISIC-2017_Test_v2_Data')
        
        images_list=sorted(os.listdir(images_path))
        images_list = [item for item in images_list if "superpixels" not in item]
        gts_list=sorted(os.listdir(gts_path))
        self.data=[]
        for i in range(len(images_list)):
            image_path=images_path+'/'+images_list[i]
            mask_path=gts_path+'/'+gts_list[i]
            self.data.append([image_path, mask_path])
        random.shuffle(self.data)
        self.transformer=transformer
        print(len(self.data))

        
            
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
    def __init__(self,mode,transformer):
        super().__init__()
        cwd=os.getcwd()
        images_path=os.path.join(cwd,'data','PH2','PH2Dataset','PH2 Dataset images')
        images_list=sorted(os.listdir(images_path))
        random.shuffle(images_list)
        self.data=[]
        for path in images_list:
            image_path=os.path.join(images_path,path,path+'_Dermoscopic_Image',path+'.bmp')
            gt_path=os.path.join(images_path,path,path+'_lesion',path+'_lesion.bmp')
            self.data.append([image_path, gt_path])
        limit=int(len(self.data)*0.8)
        if mode==TRAIN:
            self.data=self.data[:limit]
        if mode==VAL:
            self.data=self.data[limit:]
        if mode==TEST:
            self.data=self.data[limit:]
        self.transformer=transformer
        print(f'the length of datasets is {len(self.data)}')
    
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
    