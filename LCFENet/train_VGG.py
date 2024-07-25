import sys
import os
from loader import get_loader
from micro import ULTRALIGHT_VM_UNET, VGG, TRAIN, VAL, VMUNET,DAE_FORMER
from models.Model import Model
sys.path.append(os.getcwd())
from utils.loss_function import BceDiceLoss, BceIOULoss
from utils.tools import continue_train, get_logger, set_cuda,calculate_params_flops
import torch
from train_val_epoch import train_epoch,val_epoch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--datasets",
    type=str,
    default="ISIC2018",
    help="input datasets name including ISIC2017, ISIC2018, and PH2",
)
parser.add_argument(
    "--backbone",
    type=str,
    default=VGG,
    help="input VGG, VMUNet, or UltraLight_VM_UNet",
)
parser.add_argument(
    "--batchsize",
    type=int,
    default="11",
    help="input batch_size",
)
parser.add_argument(
    "--imagesize",
    type=int,
    default=256,
    help="input image resolution. 224 for VGG; 256 for VMUNet",
)
parser.add_argument(
    "--log",
    type=str,
    default="log",
    help="input log folder: ./log",
)
parser.add_argument(
    "--continues",
    type=int,
    default=0,
    help="1: continue to run; 0: don't continue to run",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default='checkpoints',
    help="the checkpoint path of last model: ./checkpoints",
)

def get_model():
    if args.backbone==VGG:
        model=Model(in_channels=[16,32,64,128,256,512],scale_factor=[1,2,4,8,16,32],model=VGG)
    if args.backbone==VMUNET:
        model=Model(in_channels=[64,128,256,256],scale_factor=[4,8,16,32],model=VMUNET)
    if args.backbone==ULTRALIGHT_VM_UNET:
        model=Model(in_channels=[8,16,32,64,128,256],scale_factor=[1,2,4,8,16,32],model=ULTRALIGHT_VM_UNET)
    if args.backbone==DAE_FORMER:
        model=Model(in_channels=[8,16,32,64,128,256],scale_factor=[4,8,16],model=DAE_FORMER)
    model = model.cuda()
    return model



def train(args):
    #init_checkpoint folder
    checkpoint_path=os.path.join(os.getcwd(),args.checkpoint,args.backbone)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    #logger
    logger = get_logger('train'+args.backbone, os.path.join(os.getcwd(),args.log))
    #initialization cuda
    set_cuda(gpu_id='4')
    #get loader
    train_loader=get_loader(args.datasets,args.batchsize,args.imagesize,mode=TRAIN)
    val_loader=get_loader(args.datasets,args.batchsize,args.imagesize,mode=VAL)
    #get model
    model=get_model()
    #calculate parameters and flops
    calculate_params_flops(model,size=args.imagesize,logger=logger)
    #set loss function
    criterion=BceDiceLoss()
    #set optim
    optimizer = torch.optim.Adam(model.parameters(), 1e-4)
    #running settings
    min_loss=1000
    start_epoch=0
    steps=0
    #Do continue to run?
    if args.continues:
        model,start_epoch,min_loss,optimizer=continue_train(model,optimizer,checkpoint_path)
        lr=optimizer.state_dict()['param_groups'][0]['lr']
        print(f'start_epoch={start_epoch},min_loss={min_loss},lr={lr}')
    #start to run the model
    for epoch in range(start_epoch,500):
        torch.cuda.empty_cache()
        #train model
        steps=train_epoch(train_loader,model,criterion,optimizer,epoch, steps,logger,save_cycles=5)
        #validate model
        loss=val_epoch(val_loader,model,criterion,epoch,logger,val_cycles=1)
        if loss<min_loss:
            print('save best.pth')
            min_loss=loss
            min_epoch=epoch
            torch.save(
            {
                'epoch': epoch,
                'min_loss': min_loss,
                'min_epoch': min_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(checkpoint_path, 'best.pth'))
        print('save last.pth')
        torch.save(
        {
            'epoch': epoch,
            'min_loss': min_loss,
            'min_epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(checkpoint_path, 'last.pth'))   

    


if __name__ == '__main__':
    args = parser.parse_args()
    train(args)
