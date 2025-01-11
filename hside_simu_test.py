import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import argparse
from utility import *
from hsi_setup import Engine, train_options, make_dataset
import time

if __name__ == '__main__':
    """Training settings"""
    parser = argparse.ArgumentParser(
        # description='Hyperspectral Image Denoising (Complex noise)')
        description='Hyperspectral Image Denoising (Gaussian noise)')
    opt = train_options(parser)
    print(opt)
    

    """Setup Engine"""
    engine = Engine(opt)

    """Dataset Setting"""
    
    # HSI2Tensor = partial(HSI2Tensor, use_2dconv=engine.get_net().use_2dconv)
    #
    #
    # target_transform = HSI2Tensor()

    """Test-Dev"""

    test_dir = opt.test_dir
    # test_dir = "./data/ICVL/icvl_test_512_gaussian/10_70/"
    # test_dir = "./data/ICVL/icvl_test_512_complex/non-iid_gaussian/"
    # test_dir = "./data/ICVL/icvl_test_512_complex/gaussian_deadline/"
    # test_dir = "./data/ICVL/icvl_test_512_complex/gaussian_impulse/"
    # test_dir = "./data/ICVL/icvl_test_512_complex/gaussian_stripe/"
    # test_dir = "./data/ICVL/icvl_test_512_complex/gaussian_mix/"

    mat_dataset = MatDataFromFolder(
        test_dir) 
    if not engine.get_net().use_2dconv:
        mat_transform = Compose([
            LoadMatHSI(input_key='input', gt_key='gt',
                    transform=lambda x:x[ ...][None], needsigma=False),
        ])
    else:
        mat_transform = Compose([
            LoadMatHSI(input_key='input', gt_key='gt', needsigma=False),
        ])

    mat_dataset = TransformDataset(mat_dataset, mat_transform)
                    

 
    mat_loader = DataLoader(
        mat_dataset,
        batch_size=1, shuffle=False,
        num_workers=1, pin_memory=opt.no_cuda
    )       

    # base_lr = opt.lr
    # epoch_per_save = 5
    # adjust_learning_rate(engine.optimizer, opt.lr)

    # engine.epoch  = 0
    

    strart_time = time.time()
    engine.test(mat_loader, test_dir)
    end_time = time.time()
    test_time = end_time-strart_time
    print('cost-time: ',(test_time/len(mat_dataset)))
