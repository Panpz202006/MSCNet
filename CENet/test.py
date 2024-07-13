import sys
import os
import time
from loader import get_loader
import imageio
from micro import CNN, MAMBA, TEST, TRANSFORMER
from models.Model import Model
sys.path.append(os.getcwd())
from utils.loss_function import BceIOULoss
from utils.tools import get_logger, set_cuda,calculate_params_flops
from train_val_epoch import train_epoch,val_epoch
import argparse
import torch
import numpy as np
from tqdm import tqdm
from utils.loss_function import adjust_lr, clip_gradient, get_metrics

parser = argparse.ArgumentParser()
parser.add_argument(
    "--datasets",
    type=str,
    default="PH2",
    help="input datasets name including ISIC2017, ISIC2018, and PH2",
)
parser.add_argument(
    "--model",
    type=str,
    default=CNN,
    help="input CNN, Mamba, or Transformer",
)

parser.add_argument(
    "--imagesize",
    type=int,
    default=224,
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
    if args.model==CNN:
        model=Model(in_channels=[64,128,256,512,512],scale_factor=[1,2,4,8,16],model=CNN)
    if args.model==MAMBA:
        model=Model(in_channels=[128,256,512,512],scale_factor=[4,8,16,32],model=MAMBA)
    if args.model==TRANSFORMER:
        model=None
    model = model.cuda()
    return model

def continue_train(model,optimizer):
    path=os.path.join(os.path.join(os.getcwd(),args.checkpoint,'last.pth'))
    if not os.path.exists(path):
        os.makedirs(path)
    loaded_data = torch.load(path)
    start_epoch=int(loaded_data['epoch'])+1
    min_loss=float(loaded_data['min_loss'])
    model.load_state_dict(loaded_data['model_state_dict'])
    optimizer.load_state_dict(loaded_data['optimizer_state_dict'])
    print('继续训练')
    return model,start_epoch,min_loss,optimizer


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
            gts.append(gt.squeeze(1).cpu().detach().numpy())
            preds.append(pred[0].squeeze(1).cpu().detach().numpy()) 
            # images=images.permute(0,2,3,1).contiguous()
            # images = images.data.cpu().detach().numpy().squeeze()
            # images = (images * 255).astype(np.uint8)

            gt_ = gt.data.cpu().detach().numpy().squeeze()
            gt_ = (gt_ * 255).astype(np.uint8)

            pred_ = pred[0].data.cpu().detach().numpy().squeeze()
            pred_ = (pred_ - pred_.min()) / (pred_.max() - pred_.min() + 1e-8)
            pred_ = (pred_ * 255).astype(np.uint8)
            # imageio.imsave(os.path.join(image_root,f'{id}.jpg'), images)
            imageio.imsave(os.path.join(gt_root,f'{id}.jpg'), gt_)
            imageio.imsave(os.path.join(pred_root,f'{id}.jpg'), pred_)
            id=id+1

    log_info=get_metrics(preds,gts)
    log_info=f'val loss={np.mean(loss_list):.4f}  {log_info}'
    print(log_info)
    logger.info(log_info)
    return np.mean(loss_list)



def test(args):
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
    criterion=BceIOULoss()
    #set optim
    optimizer = torch.optim.Adam(model.parameters(), 1e-4)
    #Do continue to run?
    model,_,_,_=continue_train(model,optimizer)
    #start to run the model
    test_epoch(test_loader,model,criterion,logger,os.path.join(os.getcwd(),'Test'))


if __name__ == '__main__':
    args = parser.parse_args()
    test(args)
