import torch
import numpy as np
import os
import imageio
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

def test_epoch(model,save_path,test_loader,criterion,logger):
    image_root =  os.path.join(save_path,'image')
    gt_root =  os.path.join(save_path,'gt','EORSSD')
    pred_root =  os.path.join(save_path,'pred','DSS','EORSSD')
    if not os.path.exists(image_root):
        os.makedirs(image_root)
    if not os.path.exists(gt_root):
        os.makedirs(gt_root)
    if not os.path.exists(pred_root):
        os.makedirs(pred_root)
    time_sum = 0
    model.eval()

    loss_list=[]
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
    log_info = f'val loss: {np.mean(loss_list):.4f}, time_sum={mean_time}'
    print(log_info)
    logger.info(log_info)
    print('存储完成')
    return np.mean(loss_list)