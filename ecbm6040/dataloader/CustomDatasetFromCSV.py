# This file includes Dataset class for PyTorch. You should first apply for the access to my google storage "hz2538",
# then use "gcsfuse --implicit-dirs" command to mount the disk on Google storage to your VM instance or local machine.

import torch
import numpy as np
import math
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import subprocess
import pandas as pd
from torch.utils.data.dataset import Dataset
import scipy.io
import nibabel as nib
from pathlib import Path


class CustomDatasetFromCSV(Dataset):
    def __init__(self, csv_path, vm_path):
        self.data = pd.read_csv(csv_path)
        self.vm_path = vm_path
        self.is_transform = True

    def transform(self, image):
        '''
        This function transforms the 3D image of np.ndarray (z,x,y) to a torch.ShortTensor (B,z,x,y).

        '''
        image_torch = torch.ShortTensor(image)
        return image_torch

    def __getitem__(self, idx):
        img_id = str(self.data.iloc[idx, 0])
        # set path according to the root of mounted disk in your VM instance or local machine
        nifti_path = self.vm_path + img_id + \
            '/unprocessed/3T/T1w_MPR1/' + img_id + '_3T_T1w_MPR1.nii.gz'
        mat_path = self.vm_path + img_id + '/unprocessed/3T/T1w_MPR1/' + img_id + '_LR.mat'
        # read nii file, and change it to numpy
        nii = nib.load(nifti_path)
        image_hr = np.array(nii.dataobj)
        # read mat file, and change it to numpy
        mat = scipy.io.loadmat(mat_path)
        image_lr = mat['out_final']
        if(self.is_transform):
            sample_lr = self.transform(image_lr)
            sample_hr = self.transform(image_hr)

        return (sample_lr, sample_hr)

    def __len__(self):
        return len(self.data)


class QiDataset(Dataset):
    def __init__(self, hr_path, lr_path):
        self.hr_path = hr_path
        self.lr_path = lr_path

        self.hr_files = sorted(list(Path(self.hr_path).glob("*.nii")))
        self.lr_files = sorted(list(Path(self.lr_path).glob("*.nii")))
        assert len(self.hr_files) == len(
            self.lr_files), "HR and LR files do not match"

    def transform(self, image):
        '''
        This function transforms the 3D image of np.ndarray (z,x,y) to a torch.ShortTensor (B,z,x,y).

        '''
        image_torch = torch.ShortTensor(image)
        return image_torch

    def __getitem__(self, idx):
        hr = nib.load(self.hr_files[idx])
        lr = nib.load(self.lr_files[idx])

        sample_lr = self.transform(np.array(lr.dataobj))
        sample_hr = self.transform(np.array(hr.dataobj))
        return (sample_lr[None, :], sample_hr[None, :])

    def __len__(self):
        return len(self.hr_files)


if __name__ == "__main__":
    ds = QiDataset("./test", "./test2")
