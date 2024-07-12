import torch
import numpy as np
import os
import imageio
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.loss_function import get_metrics


def test_epoch(model,save_path,test_loader,criterion,logger):
    image_root =  os.path.join(save_path,'image')
    gt_root =  os.path.join(save_path,'gt','PH2')
    pred_root =  os.path.join(save_path,'pred','PH2')
    if not os.path.exists(image_root):
        os.makedirs(image_root)
    if not os.path.exists(gt_root):
        os.makedirs(gt_root)
    if not os.path.exists(pred_root):
        os.makedirs(pred_root)
    time_sum = 0
    model.eval()

    loss_list=[]
    preds = []
    gtss = []

    id=1
    with torch.no_grad():
        for data in tqdm(test_loader):
            images, gts = data
            images, gts = images.cuda(non_blocking=True).float(), gts.cuda(non_blocking=True).float()
            time_start = time.time()
            pred,_,_,_,_ = model(images)
            time_end = time.time()
            time_sum = time_sum+(time_end-time_start)
            #计算损失
            loss = criterion(pred,gts)
            #计算损失
            loss_list.append(loss.item())
            gtss.append(gts.squeeze(1).cpu().detach().numpy())
            preds.append(pred.squeeze(1).cpu().detach().numpy()) 

            # images=images.permute(0,2,3,1).contiguous()
            # images = images.data.cpu().detach().numpy().squeeze()
            # images = (images * 255).astype(np.uint8)

            gts = gts.data.cpu().detach().numpy().squeeze()
            gts = (gts * 255).astype(np.uint8)

            #pred = pred.sigmoid().data.cpu().numpy().squeeze()
            pred = pred.data.cpu().detach().numpy().squeeze()
            pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
            pred = (pred * 255).astype(np.uint8)
            # imageio.imsave(os.path.join(image_root,f'{id}.jpg'), images)
            imageio.imsave(os.path.join(gt_root,f'{id}.jpg'), gts)
            imageio.imsave(os.path.join(pred_root,f'{id}.jpg'), pred)
            id=id+1
    mean_time=time_sum/(id-1)
    log_info=get_metrics(preds,gtss)
    log_info=f'val loss={np.mean(loss_list):.4f}, time_sum={mean_time},  {log_info}'
    print(log_info)
    logger.info(log_info)
    print('存储完成')
    return np.mean(loss_list)


from pickle import TRUE
import sys
import os
from Evalute.main import main_eval
from test import test_epoch
from models.Model import Model
sys.path.append(os.getcwd())
from datastes.dataset import Datasets, PH2_Datasets
from utils.loss_function import BceDiceLoss
from utils.tools import get_logger,set_seed,calculate_params_flops
import torch
from utils.transforms import Train_Transformer
from torch.utils.data import DataLoader


def main():
    print('####################文件初始化####################')
    cwd_path=os.getcwd()
    log_dir=os.path.join(cwd_path,'log')
    test_datasets_path=os.path.join(cwd_path,'data','PH2','Datasets')
    checkpoint_path=os.path.join(cwd_path,'checkpoints')
    save_path=os.path.join(cwd_path,'save')
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    print('####################初始化logger####################')
    logger = get_logger('train', log_dir)

    print('####################初始化CUDA####################')
    gpu_id='4'
    seed = 42
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    set_seed(seed)
    torch.cuda.empty_cache()

    print('####################准备模型####################')
    model=Model()
    model = model.cuda()
    calculate_params_flops(model,size=224,logger=logger)



    print('####################读取数据集####################')
    num_workers=0
    image_size=(224,224)
    test_dataset=PH2_Datasets(test_datasets_path,Train_Transformer(image_size),val=True)
    test_loader = DataLoader(test_dataset,
                              batch_size=1,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=num_workers,
                              drop_last=True)

   
    print('####################准备损失函数、优化器和学习率####################')
    criterion=BceDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), 1e-4)

    print('####################设置训练参数####################')
    min_loss=1000
    start_epoch=1

    print('####################是否继续训练####################')
    is_continue=True
    if is_continue:
        model,start_epoch,min_loss,optimizer=continue_train(model,checkpoint_path,optimizer)
    lr=optimizer.state_dict()['param_groups'][0]['lr']
    print(f'start_epoch={start_epoch},min_loss={min_loss},lr={lr}')

    print('####################是否开始测试####################')
    is_test=True
    if is_test:
        test_epoch(model=model,save_path=save_path,test_loader=test_loader,criterion=criterion,logger=logger)
        input('输入回车：继续')

def continue_train(model,checkpoint_path,optimizer):
    loaded_data = torch.load(os.path.join(checkpoint_path, 'last.pth'))
    start_epoch=int(loaded_data['epoch'])+1
    min_loss=float(loaded_data['min_loss'])
    model.load_state_dict(loaded_data['model_state_dict'])
    optimizer.load_state_dict(loaded_data['optimizer_state_dict'])
    
    print('继续训练')
    return model,start_epoch,min_loss,optimizer

if __name__ == '__main__':
    main()