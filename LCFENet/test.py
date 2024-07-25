import sys
import os
import time
from loader import get_loader
import imageio
from micro import TEST, ULTRALIGHT_VM_UNET, VGG, TRAIN, VAL, VMUNET
from models.Model import Model
sys.path.append(os.getcwd())
from utils.loss_function import BceDiceLoss, BceIOULoss
from utils.tools import continue_train, get_logger, set_cuda,calculate_params_flops
from train_val_epoch import train_epoch,val_epoch
import argparse
import torch
import numpy as np
from tqdm import tqdm
from utils.loss_function import adjust_lr, clip_gradient, get_metrics
from PIL import Image
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument(
    "--datasets",
    type=str,
    default="PH2",
    help="input datasets name including ISIC2017, ISIC2018, and PH2",
)
parser.add_argument(
    "--backbone",
    type=str,
    default=VMUNET,
    help="input VGG, VMUNet, or UltraLight_VM_UNet",
)
parser.add_argument(
    "--imagesize",
    type=int,
    default=256,
    help="input image resolution. 224 for VGG; 256 for Mamba",
)
parser.add_argument(
    "--log",
    type=str,
    default="log",
    help="input log folder: ./log",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default='checkpoints',
    help="the checkpoint path of last model: ./checkpoints",
)
parser.add_argument(
    "--testdir",
    type=str,
    default='Test',
    help="the folder is saving test results",
)

def get_model():
    if args.backbone==VGG:
        model=Model(in_channels=[16,32,64,128,256],scale_factor=[1,2,4,8,16],model=VGG)
    if args.backbone==VMUNET:
        model=Model(in_channels=[32,64,128,256],scale_factor=[4,8,16,32],model=VMUNET)
    if args.backbone==ULTRALIGHT_VM_UNET:
        model=None
    model = model.cuda()
    return model

def save_imgs(img, msk, msk_pred, id, image_root,threshold=0.5):
    img = img.squeeze(0).permute(1,2,0).detach().cpu().numpy()
    img = img / 255. if img.max() > 1.1 else img
    msk = np.where(np.squeeze(msk, axis=0) > 0.5, 1, 0)
    msk_pred = np.where(np.squeeze(msk_pred, axis=0) > threshold, 1, 0) 
    plt.figure(figsize=(7,15))

    plt.subplot(3,1,1)
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(3,1,2)
    plt.imshow(msk, cmap= 'gray')
    plt.axis('off')

    plt.subplot(3,1,3)
    plt.imshow(msk_pred, cmap = 'gray')
    plt.axis('off')

    plt.savefig(os.path.join(image_root,f'{id}.jpg'))
    plt.close()



def test_epoch(test_loader,model,criterion,logger,path):
    image_root =  os.path.join(path,'images')
    gt_root =  os.path.join(path,'gt')
    pred_root =  os.path.join(path,'pred')
    if not os.path.exists(image_root):
        os.makedirs(image_root)
    if not os.path.exists(gt_root):
        os.makedirs(gt_root)
    if not os.path.exists(pred_root):
        os.makedirs(pred_root)
    model.eval()
    loss_list=[]
    preds = []
    gts = []
    id=1
    time_sum=0
    with torch.no_grad():
        for data in tqdm(test_loader):
            images, gt = data
            images, gt = images.cuda(non_blocking=True).float(), gt.cuda(non_blocking=True).float()
            time_start = time.time()
            pred = model(images)
            time_end = time.time()
            time_sum = time_sum+(time_end-time_start)
            #计算损失
            loss = criterion(pred[0],gt)
            #计算损失
            loss_list.append(loss.item())
            save_imgs(images, 
                      gt.squeeze(1).cpu().detach().numpy(), 
                      pred[0].squeeze(1).cpu().detach().numpy(), 
                      id, image_root)

            gts.append(gt.squeeze(1).cpu().detach().numpy())
            preds.append(pred[0].squeeze(1).cpu().detach().numpy()) 
            id=id+1

    log_info=get_metrics(preds,gts)
    log_info=f'val loss={np.mean(loss_list):.4f}  {log_info}'
    print(log_info)
    logger.info(log_info)
    return np.mean(loss_list)



def test(args):
    #init_checkpoint folder
    checkpoint_path=os.path.join(os.getcwd(),args.checkpoint,args.backbone)
    #logger
    logger = get_logger('test', os.path.join(os.getcwd(),args.log))
    #initialization cuda
    set_cuda()
    #get loader
    test_loader=get_loader(args.datasets,1,args.imagesize,mode=TEST)
    #get model
    model=get_model()
    #calculate parameters and flops
    calculate_params_flops(model,size=args.imagesize,logger=logger)
    #set loss function
    criterion=BceDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), 1e-4)
    #Do continue to run?
    model,_,_,_=continue_train(model=model,optimizer=optimizer,checkpoint_path=checkpoint_path)
    #start to run the model
    test_epoch(test_loader,model,criterion,logger,os.path.join(os.getcwd(),'Test',args.backbone))


if __name__ == '__main__':
    args = parser.parse_args()
    test(args)
