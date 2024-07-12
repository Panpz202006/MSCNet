from pickle import TRUE
import sys
import os
from test import test_epoch
from models.Model import Model
sys.path.append(os.getcwd())
from datasets.dataset import Datasets
from utils.loss_function import BceDiceLoss, BceIOULoss
from utils.tools import get_logger,set_seed,calculate_params_flops
import torch
from utils.transforms import Train_Transformer
from torch.utils.data import DataLoader
from train import train_epoch,val_epoch

def main():
    print('####################文件初始化####################')
    cwd_path=os.getcwd()
    log_dir=os.path.join(cwd_path,'log')
    train_datasets_path=os.path.join(cwd_path,'data','ISIC2017','train')
    val_datasets_path=os.path.join(cwd_path,'data','ISIC2017','val')
    test_datasets_path=os.path.join(cwd_path,'data','ISIC2017','test')
    checkpoint_path=os.path.join(cwd_path,'checkpoints')
    save_path=os.path.join(cwd_path,'save')
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    print('####################初始化logger####################')
    logger = get_logger('train', log_dir)

    print('####################初始化CUDA####################')
    gpu_id='6'
    seed = 42
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    set_seed(seed)
    torch.cuda.empty_cache()

    print('####################准备模型####################')
    model=Model()
    model = model.cuda()
    calculate_params_flops(model,size=224,logger=logger)



    print('####################读取数据集####################')
    batch_size=9
    num_workers=0
    image_size=(224,224)
    train_dataset=Datasets(train_datasets_path,Train_Transformer(image_size))
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=num_workers,
                              drop_last=True)

    val_dataset=Datasets(val_datasets_path,Train_Transformer(image_size))
    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=True,
                            pin_memory=False,
                            num_workers=0,
                            drop_last=True)

    print('####################准备损失函数、优化器和学习率####################')
    criterion=BceIOULoss()
    optimizer = torch.optim.Adam(model.parameters(), 1e-4)

    print('####################设置训练参数####################')
    min_loss=1000
    min_epoch=1
    steps=0
    start_epoch=1
    end_epoch=1000
    save_cycles=10
    val_cycles=1

    print('####################是否继续训练####################')
    is_continue=True
    if is_continue:
        model,start_epoch,min_loss,optimizer=continue_train(model,checkpoint_path,optimizer)
    lr=optimizer.state_dict()['param_groups'][0]['lr']
    print(f'start_epoch={start_epoch},min_loss={min_loss},lr={lr}')

    print('####################是否开始测试####################')
    is_test=False
    if is_test:
        test_dataset=Datasets(test_datasets_path,Train_Transformer(image_size))
        test_loader = DataLoader(test_dataset,
                                batch_size=1,
                                shuffle=True,
                                pin_memory=False,
                                num_workers=0,
                                drop_last=True)
        test_epoch(model=model,save_path=save_path,test_loader=test_loader,criterion=criterion,logger=logger)
        input('输入回车：继续')
    print('####################开始训练####################')
    for epoch in range(start_epoch,end_epoch+1):
        print('训练')
        torch.cuda.empty_cache()
        steps=train_epoch(train_loader,model,criterion,optimizer,epoch, steps,logger,save_cycles=save_cycles)
        print('验证')
        loss=val_epoch(val_loader,model,criterion,epoch,logger,val_cycles=val_cycles)
        if loss<min_loss:
            print('存储best.pth')
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
        print('存储last.pth')
        torch.save(
        {
            'epoch': epoch,
            'min_loss': min_loss,
            'min_epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(checkpoint_path, 'last.pth'))        



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