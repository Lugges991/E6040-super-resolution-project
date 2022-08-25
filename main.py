from __future__ import print_function
from ecbm6040.metric.eval_metrics import ssim, psnr, nrmse
from ecbm6040.patching.patchloader import depatching
from WGAN_GP import WGAN_GP
from training_pre import training_pre
from ecbm6040.model.mDCSRN_WGAN import Discriminator
from ecbm6040.model.mDCSRN_WGAN import Generator

import os
import random
import time
import math
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.image as mpimg
from scipy import ndimage
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision
from torchvision import datasets, models, transforms
import torchvision.utils as vutils
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
# from torch.optim import lr_scheduler

from ecbm6040.dataloader.CustomDatasetFromCSV import CustomDatasetFromCSV, QiDataset

from ecbm6040.patching.patchloader import patching


# Set random seem for reproducibility
manualSeed = 999
# manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Number of workers for dataloader
workers = 1

# Batch size. It controls the number of samples once download
batch_size = 16

# Patch size, it controls the number of patches once send into the model
patch_size = 2

# The size of one image patch (eg. 64 means a cubic patch with size: 64x64x64)
cube_size = 64

# Set the usage of a patch cluster.
usage = 1.0

# Number of mDCSRN (G) pre-training steps (5e6)
num_steps_pre = 250000

# Number of WGAN training steps (1.5e7)
num_steps = 450000

# Number of WGAN D pre-training steps (1e4)
first_steps = 10000

# Learning rate for mDCSRN (G) pre-training optimizers (in paper: 1e-4)
lr_pre = 1e-4

# Learning rate for optimizers (in paper: 5e-6)
lr = 5e-6

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 2

# set GPU device
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
device = torch.device("cuda:0" if (
    torch.cuda.is_available() and ngpu > 0) else "cpu")


# Set percentage of data spliting
train_split = 0.7
validate_split = 0.1
evaluate_split = 0.1
test_split = 0.1

# Set shuffle and stablize random_seed
shuffle_dataset = True
random_seed = 999

# load data from csv
# dataset = CustomDatasetFromCSV(id_csv, vm_PATH)
# dataset_size = len(dataset)

train_hr_path = "/ptmp/lumah/data/HR/all_subj_train_crops"
train_lr_path = "/ptmp/lumah/data/LR/crops"

eval_hr_path = "/ptmp/lumah/data/HR/val"
eval_lr_path = "/ptmp/lumah/data/LR/val"

train_dataset = QiDataset(train_hr_path, train_lr_path)
eval_dataset = QiDataset(eval_hr_path, eval_lr_path)


train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=workers)


validation_loader = torch.utils.data.DataLoader(dataset=eval_dataset,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=workers)


dataloaders = {'train': train_loader, 'val': validation_loader}
dataset_sizes = {'train': len(train_dataset), 'val': len(eval_dataset)}

# Use for WGAN-training (we want to have more frequent validation)


def chunks(arr, m):
    '''
    This function split the list into m fold.
    '''
    n = int(np.floor(len(arr) / float(m)))
    arr_split = [arr[i:i + n] for i in range(0, len(arr), n)]
    return arr_split


# Split indices
# train_indices_split = chunks(train_indices, 10)
# val_indices_split = chunks(val_indices, 10)
#
# dataloaders = {'train': [], 'val': []}
# dataset_sizes = {'train': [], 'val': []}
# for i in range(10):
#     train_sampler = SubsetRandomSampler(train_indices_split[i])
#     valid_sampler = SubsetRandomSampler(val_indices_split[i])
#     train_loader = torch.utils.data.DataLoader(dataset=dataset,
#                                                batch_size=batch_size,
#                                                sampler=train_sampler,
#                                                shuffle=False,
#                                                num_workers=workers)
#     validation_loader = torch.utils.data.DataLoader(dataset=dataset,
#                                                     batch_size=batch_size,
#                                                     sampler=valid_sampler,
#                                                     shuffle=False,
#                                                     num_workers=workers)
#     dataloaders['train'].append(train_loader)
#     dataloaders['val'].append(validation_loader)
#     dataset_sizes['train'].append(len(train_indices_split[i]))
#     dataset_sizes['val'].append(len(val_indices_split[i]))


# custom weights initialization called on netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('LayerNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# Create the generator
netG = Generator(ngpu).cuda(device)
# Print the model
print(netG)
print('# generator parameters:', sum(param.numel()
      for param in netG.parameters()))


# Create the Discriminator
netD = Discriminator(ngpu).cuda(device)

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netD.apply(weights_init)

# Print the model
print(netD)
print('# discriminator parameters:', sum(param.numel()
      for param in netD.parameters()))


# Initialize Loss functions
# Supervised criterion
supervised_criterion = nn.L1Loss()
# cross-entropy loss for Discriminator alone.
D_criterion = nn.BCELoss()
# We move wasserstein Loss into the training function.


model = training_pre(netG, dataloaders,
                     dataset_sizes, supervised_criterion,
                     device, ngpu, max_step=num_steps_pre,
                     lr=lr_pre, patch_size=patch_size)


wgan_gp = WGAN_GP(netG, netD, supervised_criterion,
                  D_criterion, device, ngpu, lr=lr)
model_G, model_D = wgan_gp.training(dataloaders, max_step=num_steps, first_steps=first_steps,
                                    patch_size=patch_size)


# f = open('loss_history/train_loss_history.txt', 'rb')
# train_loss = pickle.load(f)
# f.close()
# f = open('loss_history/train_loss_D_history.txt', 'rb')
# train_D_loss = pickle.load(f)
# f.close()
# f = open('loss_history/val_loss_history.txt', 'rb')
# val_loss = pickle.load(f)
# f.close()
# f = open('loss_history/val_loss_D_history.txt', 'rb')
# val_D_loss = pickle.load(f)
# f.close()
# plt.figure(figsize=(10, 5))
# plt.title("Generator and Discriminator Loss During Training")
# plt.plot(train_loss[20000:], label="G")
# plt.plot(train_D_loss[20000:], label="D")
# plt.xlabel("iterations")
# plt.ylabel("Loss")
# plt.legend()
# plt.show()
# plt.figure(figsize=(10, 5))
# plt.title("Generator and Discriminator Loss During Validation")
# plt.plot(val_loss[1000:], label="G")
# plt.plot(val_D_loss[1000:], label="D")
# plt.xlabel("iterations")
# plt.ylabel("Loss")
# plt.legend()
# plt.show()
#
#
# def check_result_images(step, slice=50):
#     f = open('example_images/example_lr_step{}.txt'.format(step), 'rb')
#     lr_patches = pickle.load(f)
#     f.close()
#     f = open('example_images/example_sr_step{}.txt'.format(step), 'rb')
#     sr_patches = pickle.load(f)
#     f.close()
#     f = open('example_images/example_hr_step{}.txt'.format(step), 'rb')
#     hr_patches = pickle.load(f)
#     f.close()
#     f = plt.figure(figsize=(16, 8))
#     patch_size = lr_patches.shape[0]
#     for patch in range(patch_size):
#         sp = f.add_subplot(patch_size, 3, patch*3+1)
#         sp.axis('Off')
#         sp.set_title('Low resolution patch', fontsize=16)
#         plt.imshow(lr_patches[patch, 0, slice, :, :], cmap='gray')
#
#         sp = f.add_subplot(patch_size, 3, patch*3+2)
#         sp.axis('Off')
#         sp.set_title('Super resolution patch', fontsize=16)
#         plt.imshow(sr_patches[patch, 0, slice, :, :], cmap='gray')
#
#         sp = f.add_subplot(patch_size, 3, patch*3+3)
#         sp.axis('Off')
#         sp.set_title('High resolution patch', fontsize=16)
#         plt.imshow(hr_patches[patch, 0, slice, :, :], cmap='gray')
#
#
# check_result_images(step=250000, slice=40)
#
# f = open('example_images/image_lr_step448000.txt', 'rb')
# lr_image = pickle.load(f)
# f.close()
# fig = plt.figure(figsize=(16, 16))
# sp = fig.add_subplot(3, 1, 1)
# sp.axis('Off')
# sp.set_title('Low resolution image', fontsize=16)
# lr_show = ndimage.rotate(lr_image[0, :, 100, :], 90)
# plt.imshow(lr_show, cmap='gray')
# f = open('example_images/image_sr_step448000.txt', 'rb')
# sr_image = pickle.load(f)
# f.close()
# sp = fig.add_subplot(3, 1, 2)
# sp.axis('Off')
# sp.set_title('Super resolution image', fontsize=16)
# sr_show = ndimage.rotate(sr_image[0, :, 100, :], 90)
# plt.imshow(sr_show, cmap='gray')
# f = open('example_images/image_hr_step448000.txt', 'rb')
# hr_image = pickle.load(f)
# f.close()
# sp = fig.add_subplot(3, 1, 3)
# sp.axis('Off')
# sp.set_title('High resolution image', fontsize=16)
# hr_show = ndimage.rotate(hr_image[0, :, 100, :], 90)
# plt.imshow(hr_show, cmap='gray')
#
#
# f = open('example_images/image_lr_step448000.txt', 'rb')
# lr_image = pickle.load(f)
# f.close()
# fig = plt.figure(figsize=(16, 16))
# sp = fig.add_subplot(3, 1, 1)
# sp.axis('Off')
# sp.set_title('Low resolution image', fontsize=16)
# lr_show = ndimage.rotate(lr_image[0, 100, :, :], 90)
# plt.imshow(lr_show, cmap='gray')
# f = open('example_images/image_sr_step448000.txt', 'rb')
# sr_image = pickle.load(f)
# f.close()
# sp = fig.add_subplot(3, 1, 2)
# sp.axis('Off')
# sp.set_title('Super resolution image', fontsize=16)
# sr_show = ndimage.rotate(sr_image[0, 100, :, :], 90)
# plt.imshow(sr_show, cmap='gray')
# f = open('example_images/image_hr_step448000.txt', 'rb')
# hr_image = pickle.load(f)
# f.close()
# sp = fig.add_subplot(3, 1, 3)
# sp.axis('Off')
# sp.set_title('High resolution image', fontsize=16)
# hr_show = ndimage.rotate(hr_image[0, 100, :, :], 90)
# plt.imshow(hr_show, cmap='gray')
#
#
# wgan_gp = WGAN_GP(netG, netD, supervised_criterion, D_criterion, device, ngpu)
# wgan_gp.test(test_loader, patch_size=patch_size,
#              pretrainedG='models/pretrained_G_step250000', pretrainedD=' ')
# wgan_gp = WGAN_GP(netG, netD, supervised_criterion, D_criterion, device, ngpu)
# wgan_gp.test(test_loader, patch_size=patch_size,
#              pretrainedG='models/WGAN_G_step442000', pretrainedD='models/WGAN_D_step442000')
#
# wgan_gp = WGAN_GP(netG, netD, supervised_criterion, D_criterion, device, ngpu)
# wgan_gp.test(test_loader, patch_size=patch_size,
#              pretrainedG='models/final_model_G', pretrainedD='models/final_model_D')
