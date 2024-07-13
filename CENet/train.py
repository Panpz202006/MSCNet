import sys
import os
from loader import get_loader
from micro import CNN, MAMBA, TRAIN, TRANSFORMER, VAL
from models.Model import Model
sys.path.append(os.getcwd())
from utils.loss_function import BceDiceLoss, BceIOULoss
from utils.tools import get_logger, set_cuda,calculate_params_flops
import torch
from train_val_epoch import train_epoch,val_epoch
import argparse

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
    "--batchsize",
    type=int,
    default="10",
    help="input batch_size",
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
    "--continues",
    type=int,
    default=1,
    help="1: continue to run; 0: don't continue to run",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default='checkpoints',
    help="the checkpoint path of last model: ./checkpoints",
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
    path=os.path.join(os.getcwd(),args.checkpoint,'last.pth')
    if not os.path.exists(path):
        os.makedirs(path)
    loaded_data = torch.load(path)
    start_epoch=int(loaded_data['epoch'])+1
    min_loss=float(loaded_data['min_loss'])
    model.load_state_dict(loaded_data['model_state_dict'])
    optimizer.load_state_dict(loaded_data['optimizer_state_dict'])
    print('继续训练')
    return model,start_epoch,min_loss,optimizer


def train(args):
    #logger
    logger = get_logger('train', os.path.join(os.getcwd(),args.log))
    #initialization cuda
    set_cuda()
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
        model,start_epoch,min_loss,optimizer=continue_train(model,optimizer)
        lr=optimizer.state_dict()['param_groups'][0]['lr']
        print(f'start_epoch={start_epoch},min_loss={min_loss},lr={lr}')
    #start to run the model
    for epoch in range(start_epoch,400):
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
            }, os.path.join(os.path.join(os.getcwd(),args.checkpoint), 'best.pth'))
        print('save last.pth')
        torch.save(
        {
            'epoch': epoch,
            'min_loss': min_loss,
            'min_epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(os.path.join(os.getcwd(),args.checkpoint), 'last.pth'))   

    


if __name__ == '__main__':
    args = parser.parse_args()
    train(args)
