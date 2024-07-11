import numpy as np
from torchvision import transforms
import torch
import random
import torchvision.transforms.functional as TF
from PIL import ImageEnhance,Image

class Normalize:
    def __init__(self, train=True):
        if train:
            self.mean = 157.561
            self.std = 26.706
        else:
            self.mean = 157.561
            self.std = 26.706
            
    def __call__(self, data):
        img, gt = data
        img_normalized = (img-self.mean)/self.std
        img_normalized = ((img_normalized - np.min(img_normalized))  / (np.max(img_normalized)-np.min(img_normalized))) * 255.
        return img_normalized, gt


class Resize:
    def __init__(self, image_size=(256,256)):
        self.h,self.w = image_size

    def __call__(self, data):
        image, gt = data
        return TF.resize(image, [self.h, self.w]), TF.resize(gt, [self.h, self.w])




class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, data):
        image, gt = data
        if random.random() < self.prob:
            image,gt = TF.hflip(image), TF.hflip(gt)
        return image, gt


class RandomVerticalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, data):
        image, gt = data
        if random.random() < self.prob:
            image,gt = TF.vflip(image), TF.vflip(gt)
        return image, gt
    
class RandomRotate:
    def __init__(self,prob=0.5,degree=[-15,15]):
        self.prob=prob
        self.degree=degree
        self.angle = random.uniform(degree[0], degree[1])

    def __call__(self, data):
        image, gt = data
        if random.random() < self.prob:
            return TF.rotate(image,self.angle), TF.rotate(gt,self.angle)
        return image,gt


class ToTensor:
    def __init__(self):
        pass

    def __call__(self, data):
        image, gt = data
        return torch.tensor(image), torch.tensor(gt)

def Train_Transformer(image_size):
    train_transformer = transforms.Compose([
        Normalize(train=True),
        ToTensor(),
        RandomRotate(prob=0.3),
        RandomVerticalFlip(prob=0.4),
        RandomHorizontalFlip(prob=0.4),
        Resize(image_size)
    ])
    return train_transformer

def Test_Transformer(image_size):
    test_transformer = transforms.Compose([
        Normalize(train=False),
        ToTensor(),
        Resize(image_size)
    ])
    return test_transformer