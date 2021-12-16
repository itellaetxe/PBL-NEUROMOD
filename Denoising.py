# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 12:54:04 2021

@author: IMANOL
"""

import os
import numpy as np
from dipy.io.image import load_nifti
from dipy.denoise.noise_estimate import estimate_sigma
from dipy.denoise.nlmeans import nlmeans
from dipy.segment.mask import median_otsu


def NL_means(data, mask):
    """
    Parameters
    ----------
    data : 3D array
        Imported Nifti data volume where the indexes state [rows, cols, num_of_slices].
        DTI or MPRAGE files are accepted.
    mask : 3D array
        Generated 3D volume. Could be calculated using Otsu's Median method, or using
        constant numerical values.

    Returns
    -------
    output : 3D array
        Filtered data file.
    """
    sigma = estimate_sigma(data, N=32)
    output = nlmeans(data, sigma=sigma, mask=mask, patch_radius=1, block_radius=2, rician=True)
    
    return output
    
    
def generate_mask3D(data, mode):
    """
    Parameters
    ----------
    data : 3D array
        Imported Nifti data volume where the indexes state [rows, cols, num_of_slices].
        DTI or MPRAGE files are accepted.
    mode : string
        Defines whether the mask is generated using Otsu's method or manually selecting a
        threshold.

    Returns
    -------
    mask : 3D array
        Generated 3D volume. Could be calculated using Otsu's Median method, or using
        constant numerical values.
    """
    if mode == 'otsu':
        _, mask = median_otsu(data)
    elif mode == 'constant':
        mask = data > 30
        
    return mask


def signal2noise(a, axis=None, ddof=0):
    """
    Recovered from https://stackoverflow.com/questions/63177236/how-to-calculate-signal-to-noise-ratio-using-python
    """
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    
    return np.where(sd == 0, 0, m/sd)


#%% Filtering every Nifty file to improve their SNR

mydatadir = "C:\\Users\\IMANOL\\Desktop\\MU\\941_S_6854\\Accelerated_Sagittal_MPRAGE\\2020-02-14_11_53_08.0\\S924969\\example.nii"

data, _, _ = load_nifti(mydatadir, return_img=True) # affine and img are not necessary to filter data
data = np.squeeze(data, axis=3) # Delete empty 4th dimension

SNR_raw = []
for sl in range(0, data.shape[2], 1):
    SNR_tmp = signal2noise(data[:, :, sl], axis=None, ddof=0)
    SNR_raw.append(float(SNR_tmp))
    
mask = generate_mask3D(data, 'constant')
filtered_data = NL_means(data, mask)

SNR_filtered = []
for sl in range(0, filtered_data.shape[2], 1):
    SNR_tmp = signal2noise(filtered_data[:, :, sl], axis=None, ddof=0)
    SNR_filtered.append(float(SNR_tmp))
    
SNR_improvement = [m - n for m, n in zip(SNR_raw, SNR_filtered)]
print("Mean SNR of the data has been improved {} points.".format(np.mean(SNR_improvement)))
    
#plt.savefig('denoised.png', bbox_inches='tight')