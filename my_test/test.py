import os
from itertools import product
from os.path import join

import h5py
import numpy as np
import torch
from PIL import Image
from scipy.io import loadmat, savemat
from skimage import io

from utility import LMDBDataset, MatDataFromFolder, AddNoiseBlindv2, DataLoaderVal, load_tif_img

# train_data = MatDataFromFolder("/home2/szl/sert/data/ICVL/icvl_test_512_gaussian/10")
# print(train_data.__getitem__(0)['input'].shape)

# data = loadmat("/home2/szl/end6_groundTruth.mat")
# print(data)


# icvl_64_31_dir ='/home2/szl/sert/data/ICVL/train_ICVL64_31.db/'
# icvl_64_31 = LMDBDataset(icvl_64_31_dir)
# print(icvl_64_31.__getitem__(0).shape)


# filepath = "/home2/szl/sert/data/ICVL/icvl_test_512_complex/gaussian_mix/rmt_0328-1241-1.mat"
#
# mat = loadmat(filepath)
# gt = mat['gt']
# inp = mat['input']
# try:
#     # Image.fromarray(np.array(gt * 255, np.uint8)[:, :, 20]).save('/home2/szl/sert/data/ICVL/icvl_test_512_complex/gt.png')
#     Image.fromarray(np.array(inp * 255, np.uint8)[:, :, 20]).save('/home2/szl/sert/data/ICVL/icvl_test_512_complex/inp.png')
# except Exception as e:
#         print(e)

# data = np.random.rand(10, 10, 10)
# ksizes = [3, 3, 3]
# strides = [1, 1, 1]
# dshape = data.shape
# PatNum = lambda l, k, s: (np.floor((l - k) / s) + 1)
#
# TotalPatNum = 1
# for i in range(len(ksizes)):
#     TotalPatNum = TotalPatNum * PatNum(dshape[i], ksizes[i], strides[i])
#
# V = np.zeros([int(TotalPatNum)] + ksizes)   # create D+1 dimension volume
#
# args = [range(kz) for kz in ksizes]
# for s in product(*args):
#     s1 = (slice(None),) + s
#     s2 = tuple([slice(key, -ksizes[i] + key + 1 or None, strides[i]) for i, key in enumerate(s)])
#     V[s1] = np.reshape(data[s2], -1)
# print(V.shape)


import torch
import models

if __name__ == '__main__':
    # net = models.__dict__['sert_base']()
    # checkpoint = torch.load("/home2/szl/sert/checkpoints/icvl_gaussian.pth")
    # net.load_state_dict(checkpoint['net'])
    # img = loadmat("/home2/szl/sert/data/ICVL/icvl_test_512_gaussian/50/Lehavim_0910-1708.mat")
    # noisy_img = img['input']
    # clean_img = img['gt']
    # temp = noisy_img.transpose(2, 0, 1)
    # temp = torch.Tensor(temp[None, ...])
    # print(temp.shape)
    # result_img = net(temp)
    # result_img = result_img.detach().numpy()[0].transpose(1,2,0)
    # try:
    #     Image.fromarray(np.array(noisy_img * 255, np.uint8)[:, :, 27]).save('/home2/szl/noisy_img.png')
    #     Image.fromarray(np.array(clean_img * 255, np.uint8)[:, :, 27]).save('/home2/szl/clean_img.png')
    #     Image.fromarray(np.array(result_img * 255, np.uint8)[:, :, 27]).save('/home2/szl/denoise_img.png')
    # except Exception as e:
    #     print(e)


    net = models.__dict__['sert_real']()
    checkpoint = torch.load("/home2/szl/sert/checkpoints/real_realistic.pth")
    net.load_state_dict(checkpoint['net'])

    clean = "/home2/szl/sert/test_real/gt/5.tif"
    noisy = "/home2/szl/sert/test_real/input50/5.tif"
    clean_img = torch.from_numpy(np.float32(load_tif_img(clean)))
    noisy_img = torch.from_numpy(np.float32(load_tif_img(noisy)))
    ps = 512
    r = noisy_img.shape[1] // 2 - ps // 2
    c = noisy_img.shape[2] // 2 - ps // 2
    clean_img = clean_img[:, r:r + ps, c:c + ps]
    noisy_img = noisy_img[None, :, r:r + ps, c:c + ps] * 50

    clean_img = torch.clamp(clean_img, 0, 1)
    noisy_img = torch.clamp(noisy_img, 0, 1)

    result_img = net(noisy_img)
    result_img = result_img.detach().numpy()[0].transpose(1,2,0)

    clean_img = np.array(clean_img).transpose(1,2,0)
    noisy_img = np.array(noisy_img[0]).transpose(1,2,0)
    try:
        Image.fromarray(np.array(clean_img * 255, np.uint8)[:, :, 11]).save('/home2/szl/real_clean_img2.png')
        Image.fromarray(np.array(noisy_img * 255, np.uint8)[:, :, 11]).save('/home2/szl/real_noisy_img2.png')
        Image.fromarray(np.array(result_img * 255, np.uint8)[:, :, 11]).save('/home2/szl/real_denoise_img2.png')
    except Exception as e:
        print(e)
