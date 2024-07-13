import torch
import numpy as np
from tqdm import tqdm
from utils.loss_function import adjust_lr, clip_gradient, get_metrics



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
            pred = model(images)
            #计算损失
            loss = criterion(pred[0],gt)
            #计算损失
            loss_list.append(loss.item())
            gts.append(gt.squeeze(1).cpu().detach().numpy())
            preds.append(pred[0].squeeze(1).cpu().detach().numpy()) 
    log_info=get_metrics(preds,gts)
    log_info=f'val loss={np.mean(loss_list):.4f}  {log_info}'
    print(log_info)
    logger.info(log_info)
    return np.mean(loss_list)


def train_epoch(train_loader,model,criterion,optimizer,epoch,steps,logger,save_cycles=5):
    model.train()
    loss_list=[]
    for step,data in enumerate(train_loader):
        steps+=step
        #清空梯度信息
        optimizer.zero_grad()
        images, gts = data
        images, gts = images.cuda(non_blocking=True).float(), gts.cuda(non_blocking=True).float()
        pred=model(images)
        loss=criterion(pred[0],gts)
        for i in range(1,len(pred)):
            loss+=criterion(pred[i],gts)
        loss.backward()
        clip_gradient(optimizer, 0.5)
        optimizer.step()
        loss_list.append(loss.item())
        if step%save_cycles==0:
            lr=optimizer.state_dict()['param_groups'][0]['lr']
            log_info=f'train: epoch={epoch}, step={step}, loss={np.mean(loss_list):.4f}, lr={lr:.7f}'
            print(log_info)
            logger.info(log_info)
    adjust_lr(optimizer, epoch, init_lr=1e-4, decay_rate=0.1, decay_epoch=50)
    return step