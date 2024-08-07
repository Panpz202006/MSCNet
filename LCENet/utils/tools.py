import logging.handlers
import os
import logging
import numpy as np
import random
import torch
import torch.backends.cudnn as cudnn
from matplotlib import pyplot as plt

def get_logger(name, log_dir):
    '''
    Args:
        name(str): name of logger
        log_dir(str): path of log
    '''
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    info_name = os.path.join(log_dir, '{}.info.log'.format(name))
    info_handler = logging.handlers.TimedRotatingFileHandler(info_name,
                                                             when='D',
                                                             encoding='utf-8')
    info_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    info_handler.setFormatter(formatter)
    logger.addHandler(info_handler)

    return logger


def set_seed(seed):
    # for hash
    os.environ['PYTHONHASHSEED'] = str(seed)
    # for python and numpy
    random.seed(seed)
    np.random.seed(seed)
    # for cpu gpu
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # for cudnn
    cudnn.benchmark = False
    cudnn.deterministic = True



def get_optimizer(config, model):
    return torch.optim.AdamW(
            model.parameters(),
            lr = config['lr'],
            betas = config['betas'],
            eps = config['eps'],
            weight_decay = config['weight_decay'],
            amsgrad = config['amsgrad']
        )


def get_scheduler(config, optimizer):
    '''
    return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max = config.T_max,
            eta_min = config.eta_min,
            last_epoch = config.last_epoch
        )
    '''
    return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['step_size'],
            gamma=config['gamma']
        )


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




import torch
from thop import profile
def calculate_params_flops(model,size=480,logger=None):
    input = torch.randn(1, 3, size, size).cuda()
    flops, params = profile(model, inputs=(input,))
    print('flops',flops/1e9)			## 打印计算量
    print('params',params/1e6)			## 打印参数量
    total = sum(p.numel() for p in model.parameters())
    print("Total params: %.2fM" % (total/1e6))
    logger.info(f'flops={flops/1e9}, params={params/1e6}, Total params≈{total/1e6:.2f}M')


def set_cuda(gpu_id='6'):
    seed = 42
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    set_seed(seed)
    torch.cuda.empty_cache()

def continue_train(model,optimizer,checkpoint_path):
    path=os.path.join(checkpoint_path,'best.pth')
    if not os.path.exists(path):
        os.makedirs(path)
    print(path)
    loaded_data = torch.load(path)
    start_epoch=int(loaded_data['epoch'])+1
    min_loss=float(loaded_data['min_loss'])
    model.load_state_dict(loaded_data['model_state_dict'])
    optimizer.load_state_dict(loaded_data['optimizer_state_dict'])
    print('继续训练')
    return model,start_epoch,min_loss,optimizer

def continue_test(model,checkpoint_path):
    path=os.path.join(checkpoint_path,'best.pth')
    if not os.path.exists(path):
        os.makedirs(path)
    print(path)
    loaded_data = torch.load(path)
    start_epoch=int(loaded_data['epoch'])+1
    min_loss=float(loaded_data['min_loss'])
    model.load_state_dict(loaded_data['model_state_dict'])
    print('继续训练')
    return model,start_epoch,min_loss