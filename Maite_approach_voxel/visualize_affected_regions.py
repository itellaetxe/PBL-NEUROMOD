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
bide_fa_dual = "C:/Users/Iñigo/PycharmProjects/PBL-NEUROMOD/Maite_approach_voxel/FA_dual_regs_FDR_0p05_L1O.txt"
FA_dual_regs = pd.read_csv(bide_fa_dual, header=None)
FA_dual_nifti = atlas
FA_dual_regs = FA_dual_regs.iloc[:, 0].tolist()
# Look for the interesting regions
tru_idxs = LUT["parcel"].str.contains('|'.join(FA_dual_regs))
values_to_one = LUT["value"][tru_idxs].tolist()
# Mask for found regions
bool_mask = ~np.isin(atlas, values_to_one)
# mask_idxs = np.argwhere(bool_mask)
FA_dual_nifti[bool_mask] = 0
# Save nifti
# save_nifti("E:/PBL_MASTER/REGION_NIFTI/FA_dual_region_nifti.nii.gz", FA_dual_nifti, atlas_affine)

#############################
# Load atlas
atlas, atlas_affine = load_nifti("E:/PBL_MASTER/BN_Atlas_246_1mm.nii.gz")
# Load FA_indiv_regions
bide_fa_indiv = "C:/Users/Iñigo/PycharmProjects/PBL-NEUROMOD/Maite_approach_voxel/FA_indiv_regs_FDR_0p05_L1O.txt"
FA_indiv_regs = pd.read_csv(bide_fa_indiv, header=None)
FA_indiv_nifti = atlas
FA_indiv_regs = FA_indiv_regs.iloc[:, 0].tolist()
# Look for the interesting regions
tru_idxs = LUT["parcel"].str.contains('|'.join(FA_indiv_regs))
values_to_one = LUT["value"][tru_idxs].tolist()
# Mask for found regions
bool_mask = ~np.isin(atlas, values_to_one)
FA_indiv_nifti[bool_mask] = 0
# Save nifti
save_nifti("E:/PBL_MASTER/REGION_NIFTI/FA_indiv_region_nifti.nii.gz", FA_indiv_nifti, atlas_affine)

#############################
# Load atlas
atlas, atlas_affine = load_nifti("E:/PBL_MASTER/BN_Atlas_246_1mm.nii.gz")
# Load MD_dual_regions
bide_md_dual = "C:/Users/Iñigo/PycharmProjects/PBL-NEUROMOD/Maite_approach_voxel/MD_dual_regs_FDR_0p05_L1O.txt"
MD_dual_regs = pd.read_csv(bide_md_dual, header=None)
MD_dual_nifti = atlas
MD_dual_regs = MD_dual_regs.iloc[:, 0].tolist()
# Look for the interesting regions
tru_idxs = LUT["parcel"].str.contains('|'.join(MD_dual_regs))
values_to_one = LUT["value"][tru_idxs].tolist()
# Mask for found regions
bool_mask = ~np.isin(atlas, values_to_one)
MD_dual_nifti[bool_mask] = 0
# Save nifti
save_nifti("E:/PBL_MASTER/REGION_NIFTI/MD_dual_region_nifti.nii.gz", MD_dual_nifti, atlas_affine)

#############################
# Load atlas
atlas, atlas_affine = load_nifti("E:/PBL_MASTER/BN_Atlas_246_1mm.nii.gz")
# Load MD_indiv_regions
bide_md_indiv = "C:/Users/Iñigo/PycharmProjects/PBL-NEUROMOD/Maite_approach_voxel/MD_indiv_regs_FDR_0p05_L1O.txt"
MD_indiv_regs = pd.read_csv(bide_md_indiv, header=None)
MD_indiv_nifti = atlas
MD_indiv_regs = MD_indiv_regs.iloc[:, 0].tolist()
# Look for the interesting regions
tru_idxs = LUT["parcel"].str.contains('|'.join(MD_indiv_regs))
values_to_one = LUT["value"][tru_idxs].tolist()
# Mask for found regions
bool_mask = ~np.isin(atlas, values_to_one)
MD_indiv_nifti[bool_mask] = 0
# Save nifti
save_nifti("E:/PBL_MASTER/REGION_NIFTI/MD_indiv_region_nifti.nii.gz", MD_indiv_nifti, atlas_affine)


