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
            pred,_,_,_,_ = model(images)
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


def train_epoch(train_loader,model,criterion,optimizer,epoch, steps,logger,save_cycles=5):
    model.train()
    loss_list=[]
    for step,data in enumerate(train_loader):
        steps+=step
        #清空梯度信息
        optimizer.zero_grad()
        images, gts = data
        images, gts = images.cuda(non_blocking=True).float(), gts.cuda(non_blocking=True).float()
        #得到显著性预测图
        pred1,pred2,pred3,pred4,pred5 = model(images)
        #计算损失
        loss1 = criterion(pred1,gts)
        loss2 = criterion(pred2,gts)
        loss3 = criterion(pred3,gts)
        loss4 = criterion(pred4,gts)
        loss5 = criterion(pred5,gts)
        loss = loss1+loss2+loss3+loss4+loss5
        #反向传播
        loss.backward()
        clip_gradient(optimizer, 0.5)

        #更新参数
        optimizer.step()
        loss_list.append(loss.item())
        if step%save_cycles==0:
            lr=optimizer.state_dict()['param_groups'][0]['lr']
            log_info=f'train: epoch={epoch}, step={step}, loss={np.mean(loss_list):.4f}, lr={lr:.4f},loss1={loss1.item():.4f},loss2={loss2.item():.4f},loss3={loss3.item():.4f},loss4={loss4.item():.4f},loss5={loss5.item():.4f}'
            print(log_info)
            logger.info(log_info)
    adjust_lr(optimizer, epoch, init_lr=1e-4, decay_rate=0.1, decay_epoch=50)

    return step
