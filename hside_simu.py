import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import argparse
from utility import *
import datetime
import time
from hsi_setup import Engine, train_options, make_dataset
import wandb

"""
在ICVL数据集上处理高斯噪声
"""
if __name__ == '__main__':
    """Training settings"""
    

    parser = argparse.ArgumentParser(
    description='Hyperspectral Image Denoising (Gaussian noise)')
    opt = train_options(parser)
    print(opt)

    data = datetime.datetime.now()
    wandb.init(project="hsi-denoising", entity="name",name=opt.arch+opt.prefix+'-'+str(data.month)+'-'+str(data.day)+'-'+str(data.hour)+':'+str(data.minute),config=opt)
    wandb.config.update(parser)

    """Setup Engine"""
    engine = Engine(opt)


    """Dataset Setting"""
    
    HSI2Tensor = partial(HSI2Tensor, use_2dconv=engine.get_net().use_2dconv)

    target_transform = HSI2Tensor()

    train_transform = Compose([
        # 添加高斯噪声
        AddNoiseBlindv1(10,70),
        HSI2Tensor()
    ])

    icvl_64_31_dir ='./data/ICVL/train_ICVL64_31.db/'
    icvl_64_31 = LMDBDataset(icvl_64_31_dir)
    
    target_transform = HSI2Tensor()
    # 构建测试数据集 -- 格式：C H W
    train_dataset = ImageTransformDataset(icvl_64_31, train_transform,target_transform)
    print('==> Preparing data..')

    """Test-Dev"""
    basefolder = './data/ICVL/icvl_val_512_gaussian/10_70/'

    mat_datasets = [MatDataFromFolder(basefolder, size=5)]

    if not engine.get_net().use_2dconv:
        mat_transform = Compose([
            LoadMatHSI(input_key='input', gt_key='gt',
                    transform=lambda x:x[ ...][None], needsigma=False),
        ])
    else:
        mat_transform = Compose([
            LoadMatHSI(input_key='input', gt_key='gt', needsigma=False),
        ])

    mat_datasets = [TransformDataset(mat_dataset, mat_transform)
                    for mat_dataset in mat_datasets]

    train_loader = DataLoader(train_dataset,
                              batch_size=opt.batchSize, shuffle=True,
                              num_workers=opt.threads, pin_memory=not opt.no_cuda, worker_init_fn=worker_init_fn)
    
    mat_loaders = [DataLoader(
        mat_dataset,
        batch_size=1, shuffle=False,
        num_workers=1, pin_memory=opt.no_cuda
    ) for mat_dataset in mat_datasets]        

    base_lr = opt.lr
    epoch_per_save = 5
    adjust_learning_rate(engine.optimizer, opt.lr)

    # from epoch 50 to 100
    engine.epoch  = 0 
    while engine.epoch < 100:
        np.random.seed()

        if engine.epoch == 50:
            adjust_learning_rate(engine.optimizer, base_lr*0.1)
        
        engine.train(train_loader,mat_loaders[0])
        engine.validate(mat_loaders[0], 'icvl_base_10_70')

        display_learning_rate(engine.optimizer)
        print('Latest Result Saving...')
        model_latest_path = os.path.join(engine.basedir, engine.prefix, 'model_latest.pth')
        engine.save_checkpoint(
            model_out_path=model_latest_path
        )

        display_learning_rate(engine.optimizer)
        if engine.epoch % epoch_per_save == 0:
            engine.save_checkpoint()
    wandb.finish()
 
