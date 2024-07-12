import torch.nn as nn
import torch
def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def adjust_lr(optimizer, epoch, init_lr=1e-3, decay_rate=0.5, decay_epoch=50):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr*decay
        print('decay_epoch: {}, Current_LR: {}'.format(decay_epoch, init_lr*decay))


def _iou(pred, target, size_average = True):
    b = pred.shape[0]
    IoU = 0.0
    for i in range(0,b):
        #compute the IoU of the foreground
        Iand1 = torch.sum(target[i,:,:]*pred[i,:,:])  # pred 和 target 均为B*H*W
        Ior1 = torch.sum(target[i,:,:]) + torch.sum(pred[i,:,:])-Iand1
        IoU1 = Iand1/Ior1

        #IoU loss is (1-IoU1)
        IoU = IoU + (1-IoU1)

    return IoU/b

class IOU(torch.nn.Module):
    def __init__(self, size_average = True):
        super(IOU, self).__init__()
        self.size_average = size_average

    def forward(self, pred, target):

        return _iou(pred, target, self.size_average)



class BceIOULoss(nn.Module):
    def __init__(self, weight1=0.5,weight2=0.5):
        super(BceIOULoss, self).__init__()
        self.bce = BCELoss()
        self.weight1 = weight1
        self.weight2 = weight2
        self.IOU=IOU()
    
    
    def forward(self, pred, target):
        bceloss = self.bce(pred, target)
        iouloss=self.IOU(pred,target)
        loss = self.weight1 * bceloss + self.weight2*iouloss
        return loss


import numpy as np
from sklearn.metrics import confusion_matrix

def get_metrics(preds,gts):
    preds = np.array(preds).reshape(-1)
    gts = np.array(gts).reshape(-1)
    y_pre = np.where(preds>=0.5, 1, 0)
    y_true = np.where(gts>=0.5, 1, 0)
    confusion = confusion_matrix(y_true, y_pre)
    TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1] 
    precision=float(TP)/float(TP+FP) if float(TP + FP) != 0 else 0
    recall=float(TP)/float(TP+FN) if float(TP + FN) != 0 else 0
    f_beta=(1+0.3)*(precision*recall)/(0.3*precision+recall) if float(0.3*precision+recall) != 0 else 0
    e_measure=1 - (1 / (1 + 1) * ((precision * recall) / (1 * precision + recall))) if float(1 * precision + recall) != 0 else 0
    accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
    sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
    specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
    f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
    miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0
    log_info = f'miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
                specificity: {specificity}, sensitivity: {sensitivity}, precision: {precision},\
                     recall: {recall}, f_beta: {f_beta}, e_measure: {e_measure},'
   
    return log_info
    


class BCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bceloss = nn.BCELoss()

    def forward(self, pred, target):
        size = pred.size(0)
        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)

        return self.bceloss(pred_, target_)



class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1
        size = pred.size(0)

        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)
        intersection = pred_ * target_
        dice_score = (2 * intersection.sum(1) + smooth)/(pred_.sum(1) + target_.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum()/size

        return dice_loss



class BceDiceLoss(nn.Module):
    def __init__(self, wb=1, wd=1):
        super().__init__()
        self.bce = BCELoss()
        self.dice = DiceLoss()
        self.wb = wb
        self.wd = wd

    def forward(self, pred, target):
        bceloss = self.bce(pred, target)
        diceloss = self.dice(pred, target)
        loss = self.wd * diceloss + self.wb * bceloss
        return loss