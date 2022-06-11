import numpy as np
import pandas as pd
from dipy.io.image import load_nifti, save_nifti
import matplotlib.pyplot as plt
from dipy.viz import regtools

# Load atlas
atlas, atlas_affine = load_nifti("E:/PBL_MASTER/BN_Atlas_246_1mm.nii.gz")

# # Load MNI152
# mni, mni_affine = load_nifti("E:/PBL_MASTER/HCP40_MNI_1mm.nii.gz")

# Load atlas LUT
LUT = pd.read_csv("E:/PBL_MASTER/BN_Atlas_246_LUT.txt", sep=" ")
LUT = LUT.iloc[:, 0:2]
LUT.columns = ["value", "parcel"]

# Load FA_dual_regions
bide_fa_dual = "C:/Users/Iñigo/PycharmProjects/PBL-NEUROMOD/Maite_approach_voxel/FA_dual_regs_FDR_0p05.txt"
FA_dual_regs = pd.read_csv(bide_fa_dual, header=None)
FA_dual_nifti = atlas



# Load FA_indiv_regions
bide_fa_indiv = "C:/Users/Iñigo/PycharmProjects/PBL-NEUROMOD/Maite_approach_voxel/FA_indiv_regs_FDR_0p05.txt"
FA_indiv_regs = pd.read_csv(bide_fa_indiv, header=None)




