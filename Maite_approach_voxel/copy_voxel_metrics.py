import pandas as pd
import numpy as np
import os
import re
import shutil

# TRAIN AND TEST
base_path_train = "Q:/TRAIN/"
dirs = os.listdir(base_path_train)
r = re.compile("sub-")
filtered = [folder for folder in dirs if r.match(folder)]
subs = [base_path_train + sub for sub in filtered]

base_path_test_ad = "Q:/TEST/AD/"
dirs = os.listdir(base_path_test_ad)
r = re.compile("sub-")
filtered = [folder for folder in dirs if r.match(folder)]
substad = [base_path_test_ad + sub for sub in filtered]

base_path_test_cn = "Q:/TEST/NC/"
dirs = os.listdir(base_path_test_cn)
r = re.compile("sub-")
filtered = [folder for folder in dirs if r.match(folder)]
substcn = [base_path_test_cn + sub for sub in filtered]

subs.extend(substad)
subs.extend(substcn)

for sub in subs:
    subname = os.path.basename(sub)
    try:
        os.mkdir("E:/PBL_MASTER/VOXEL_METRICS" + f"/{subname}")
        # Copy FA
        shutil.copyfile(sub + f"/DTI_metrics/{subname}__fa.nii.gz",
                        "E:/PBL_MASTER/VOXEL_METRICS" + f"/{subname}/{subname}__fa.nii.gz")
        # Copy MD
        shutil.copyfile(sub + f"/DTI_metrics/{subname}__md.nii.gz",
                        "E:/PBL_MASTER/VOXEL_METRICS" + f"/{subname}/{subname}__md.nii.gz")
        # Copy T1
        shutil.copyfile(sub + f"/Register_T1/{subname}__t1_warped.nii.gz",
                        "E:/PBL_MASTER/VOXEL_METRICS" + f"/{subname}/{subname}__t1_warped.nii.gz")
    except:
        print("File or folder already exists.")

# # AD GROUP
# base_path_ad = "Q:/PROCESSED_ADNI_AD_GROUP/PROCESSED_AD_GROUP/"
# dirs = os.listdir(base_path_ad)
# r = re.compile("sub-")
# filtered = [folder for folder in dirs if r.match(folder)]
# subs = [base_path_ad + sub for sub in filtered]
#
# for sub in subs:
#     subname = os.path.basename(sub)
#     os.mkdir("E:/PBL_MASTER/AD_GROUP_VOXEL_METRICS" + f"/{subname}")
#     # Copy FA
#     shutil.copyfile(sub + f"/DTI_metrics/{subname}__fa.nii.gz",
#                     "E:/PBL_MASTER/AD_GROUP_VOXEL_METRICS" + f"/{subname}/{subname}__fa.nii.gz")
#     # Copy MD
#     shutil.copyfile(sub + f"/DTI_metrics/{subname}__md.nii.gz",
#                     "E:/PBL_MASTER/AD_GROUP_VOXEL_METRICS" + f"/{subname}/{subname}__md.nii.gz")
#     # Copy T1
#     shutil.copyfile(sub + f"/Register_T1/{subname}__t1_warped.nii.gz",
#                     "E:/PBL_MASTER/AD_GROUP_VOXEL_METRICS" + f"/{subname}/{subname}__t1_warped.nii.gz")
#
# # CN GROUP
# base_path_cn = "Q:/PROCESSED_ADNI_CONTROL_GROUP/results/"
# dirs = os.listdir(base_path_cn)
# r = re.compile("sub-")
# filtered = [folder for folder in dirs if r.match(folder)]
# subs = [base_path_ad + sub for sub in filtered]
#
# for sub in subs:
#     subname = os.path.basename(sub)
#     os.mkdir(["E:/PBL_MASTER/CN_GROUP_VOXEL_METRICS" + f"/{subname}"])
#     # Copy FA
#     shutil.copyfile(sub + f"/DTI_metrics/{subname}__fa.nii.gz",
#                     "E:/PBL_MASTER/CN_GROUP_VOXEL_METRICS" + f"/{subname}/{subname}__fa.nii.gz")
#     # Copy MD
#     shutil.copyfile(sub + f"/DTI_metrics/{subname}__md.nii.gz",
#                     "E:/PBL_MASTER/CN_GROUP_VOXEL_METRICS" + f"/{subname}/{subname}__md.nii.gz")
#     # Copy T1
#     shutil.copyfile(sub + f"/Register_T1/{subname}__t1_warped.nii.gz",
#                     "E:/PBL_MASTER/CN_GROUP_VOXEL_METRICS" + f"/{subname}/{subname}__t1_warped.nii.gz")
