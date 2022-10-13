import skimage.measure as measure
from skimage.metrics import structural_similarity, peak_signal_noise_ratio, normalized_root_mse
import torch
import numpy as np

def ssim(img_true, img_test):
    '''
    This function input two batches of true images and the fake images. Use skimage.measure.compare_ssim function to compute the mean structural similarity index between two images.
    (Input) img_true: the input should be derived from dataloader, it's in torch.ShortTensor (B,z,x,y). By default, it should be HR images.
    (Input) img_test: the input should be derived from depatching function, it's in torch.float (B,z,x,y). By default, it should be SR images.
    (Output) ssim: an ndarray with length (B,1), which contains the ssim value for each image in the batch.
    '''
    img_true = img_true.float() / 4095.0
    img_true = img_true.cpu().numpy()
    
    img_test = img_test.cpu().numpy()
    
    ssim=[]
    for i in range(img_true.shape[0]):
        ssim = np.append(ssim, structural_similarity(img_true[i], img_test[i], channel_axis=0))
    return ssim

def psnr(img_true, img_test):
    '''
    This function input two batches of true images and the fake images. Use skimage.measure.compare_psnr function to compute the peak signal to noise ratio (PSNR) between two images.
    (Input) img_true: the input should be derived from dataloader, it's in torch.ShortTensor (B,z,x,y). By default, it should be HR images.
    (Input) img_test: the input should be derived from depatching function, it's in torch.float (B,z,x,y). By default, it should be SR images.
    (Output) psnr: an ndarray with length (B,1), which contains the psnr value for each image in the batch.
    '''
    img_true = img_true.float() / 4095.0
    img_true = img_true.cpu().numpy()
    
    img_test = img_test.cpu().numpy()
    psnr=[]
    for i in range(img_true.shape[0]):
        psnr = np.append(psnr, peak_signal_noise_ratio(img_true[i], img_test[i]))
    return psnr

def nrmse(img_true, img_test):
    '''
    This function input two batches of true images and the fake images. Use skimage.measure.compare_nrmse function to compute the normalized root mean-squared error (NRMSE) between two images.
    (Input) img_true: the input should be derived from dataloader, it's in torch.ShortTensor (B,z,x,y). By default, it should be HR images.
    (Input) img_test: the input should be derived from depatching function, it's in torch.float (B,z,x,y). By default, it should be SR images.
    (Output) nrmse: an ndarray with length (B,1), which contains the psnr value for each image in the batch.
    '''
    img_true = img_true.float() / 4095.0
    img_true = img_true.cpu().numpy()
    
    img_test = img_test.cpu().numpy()
    nrmse=[]
    for i in range(img_true.shape[0]):
        nrmse = np.append(nrmse, normalized_root_mse(img_true[i], img_test[i]))
    return nrmse
