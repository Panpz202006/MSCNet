import torch
import numpy as np
from tqdm import tqdm
from utils.loss_function import get_metrics


def val_epoch(val_loader,model,criterion,epoch,logger,val_cycles=2):
    if epoch % val_cycles!=0:
        return
    model.eval()
    loss_list=[]
    preds = []
    gts = []

    with torch.no_grad():
        for data in tqdm(val_loader):
            images, gt = data
            images, gt = images.cuda(non_blocking=True).float(), gt.cuda(non_blocking=True).float()
            pred,_,_,_ = model(images)
            #计算损失
            loss = criterion(pred,gt)
            #计算损失
            loss_list.append(loss.item())
            gts.append(gt.squeeze(1).cpu().detach().numpy())
            preds.append(pred.squeeze(1).cpu().detach().numpy()) 
    log_info=get_metrics(preds,gts)
    log_info=f'val loss={np.mean(loss_list):.4f}  {log_info}'
    print(log_info)
    logger.info(log_info)
    return np.mean(loss_list)
