import os
import re
import numpy as np
import pandas as pd
from dipy.viz import regtools
from dipy.align.imaffine import (AffineMap, MutualInformationMetric, AffineRegistration)
from dipy.align.transforms import (TranslationTransform3D, RigidTransform3D, AffineTransform3D)
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.imwarp import DiffeomorphicMap
from dipy.align.metrics import CCMetric
from dipy.io.image import load_nifti_data, load_nifti, save_nifti
import matplotlib.pyplot as plt
import scipy.io


def diffeomorphic_reg(sdr, template_img, moving_img, template_affine, moving_affine, affine):
    print("--> Diffeomorphic registration started.")

    diffeom_mapping = sdr.optimize(template_img, moving_img, template_affine, moving_affine, affine.affine)

    diffeom_transformed = diffeom_mapping.transform(moving_img)
    # regtools.overlay_slices(template_img, diffeom_transformed, None, 0, "Template", "Transformed")
    print("--> Diffeomorphic registration finished.")

    return diffeom_transformed, diffeom_mapping


def affine_reg(affreg, template_img, moving_img, template_affine, moving_affine, rigid):
    print("--> Affine registration started.")

    transform_affine = AffineTransform3D()
    params0 = None
    affreg.level_iters = [1000, 1000, 100]
    affine = affreg.optimize(template_img, moving_img, transform_affine, params0, template_affine, moving_affine,
                             starting_affine=rigid.affine)

    affine_transformed = affine.transform(moving_img)
    # regtools.overlay_slices(template_img, affine_transformed, None, 0, "Template", "Transformed")
    print("--> Affine registration finished.")

    return affine_transformed, affine


def rigid_reg(affreg, template_img, moving_img, template_affine, moving_affine, translation):
    print("--> Rigid registration started.")
    transform_rigid = RigidTransform3D()
    params0 = None
    rigid = affreg.optimize(template_img, moving_img, transform_rigid, params0, template_affine, moving_affine,
                            starting_affine=translation.affine)

    rigidbody_transformed = rigid.transform(moving_img)
    # regtools.overlay_slices(template_img, rigidbody_transformed, None, 0, "Template", "Transformed")
    print("--> Rigid registration finished.")

    return rigidbody_transformed, rigid


def translation_reg(affreg, template_img, moving_img, template_affine, moving_affine):
    print("--> Translation registration started.")
    transform_translation = TranslationTransform3D()
    params0 = None

    # This starts computing the translation
    translation = affreg.optimize(template_img, moving_img, transform_translation, params0, template_affine,
                                  moving_affine)

    translated_img = translation.transform(moving_img)
    # regtools.overlay_slices(template_img, translated_img, None, 0, "Template", "Transformed")
    print("--> Translation registration finished.")

    return translated_img, translation


########################################################################################################################
########################################################################################################################
srcp = "C:\\Users\\Iñigo\\OneDrive - Mondragon Unibertsitatea\\BIOMEDIKA\\MASTER\\PBL-NEURO-DISEASE-MODELS\\try_Maite_approach"
# Load the BRAINNETOME ATLAS (constant volume)
srcbn = os.path.join(srcp, "BN_Atlas_246_1mm.nii.gz")
atlas, atlas_affine, atlas_voxsize = load_nifti(srcbn, return_voxsize=True)

# Load MNI152 Standard (constant volume)
src_mni = os.path.join(srcp, "HCP40_MNI_1mm.nii.gz")
mni, mni_affine, mni_voxsize = load_nifti(src_mni, return_voxsize=True)

# Subject by subject, perform registration and extract FA and MD mean values.
base_path = "E:/PBL_MASTER/VOXEL_METRICS/"
dirs = os.listdir(base_path)
r = re.compile("sub-")
filtered = [folder for folder in dirs if r.match(folder)]
subs = [base_path + sub for sub in filtered]

# Load Brainnetome LUT and extract names
src_LUT = os.path.join(base_path, "BN_Atlas_246_LUT.txt")
parcellations = pd.read_csv(src_LUT, sep=" ", header=None)
parcellation_names = parcellations.iloc[1:, 1].reset_index(drop=True)
parcel_vals = np.unique(atlas)

for srcp in subs:
    subname = os.path.basename(srcp)
    print(f"Starting with subject {subname}")
    # Load T1(living in DWI) volume
    src_t1 = os.path.join(srcp, f"{subname}__t1_warped.nii.gz")
    t1, t1_affine, t1_voxsize = load_nifti(src_t1, return_voxsize=True)

    # Affine registration from T1dwi to MNI152 (atlas also lives here)
    identity = np.eye(4)
    affine_map = AffineMap(identity, t1.shape, t1_affine, mni.shape, mni_affine)
    resampled = affine_map.transform(mni)
    # regtools.overlay_slices(t1, resampled, None, 0, "Template", "Moving")

    nbins = 32
    sampling_group = None
    metric = MutualInformationMetric(nbins, sampling_group)

    level_iters = [10, 10, 5]
    sigmas = [3.0, 1.0, 0.0]
    factors = [4, 2, 1]
    affreg = AffineRegistration(metric=metric, level_iters=level_iters, sigmas=sigmas, factors=factors)

    # Linear Registration
    translated_img, translation = translation_reg(affreg, t1, mni, t1_affine, mni_affine)
    rigidbody_transformed, rigid = rigid_reg(affreg, t1, mni, t1_affine, mni_affine, translation)
    affine_transformed, affine = affine_reg(affreg, t1, mni, t1_affine, mni_affine, rigid)

    # Non-linear Registration
    # Set up diffeomorphic registration
    metric = CCMetric(3)
    level_iters = [10, 10, 5]
    sdr = SymmetricDiffeomorphicRegistration(metric, level_iters)

    # Perform diffeomorphic registration
    diffeom_transformed, diffeom_mapping = diffeomorphic_reg(sdr, t1, mni, t1_affine, mni_affine,
                                                             affine)
    forward_field = diffeom_mapping.get_forward_field()
    scipy.io.savemat(srcp + "/" + subname + "_diffeo_forward_field.mat", {'forwardfield_mni_to_patient': forward_field},
                     appendmat=True)
    backward_field = diffeom_mapping.get_backward_field()
    scipy.io.savemat(srcp + "/" + subname + "_diffeo_backward_field.mat",
                     {'backward_field_mni_to_patient': backward_field}, appendmat=True)

    # Apply diffeomorphic mapping to ATLAS and check overlay :)
    atlas_warped = diffeom_mapping.transform(atlas, interpolation='nn')  # NN interp does not alter atlas values.
    # regtools.overlay_slices(diffeom_transformed, atlas_warped, None, 0, "Template", "Moving")
    # plt.show()  # The atlas did move to the T1dwi space correctly :)
    # Load the FA volume

    src_fa = os.path.join(srcp, f"{subname}__fa.nii.gz")
    fa, fa_affine = load_nifti(src_fa)

    # Load the MD volume
    src_md = os.path.join(srcp, f"{subname}__md.nii.gz")
    md, md_affine = load_nifti(src_md)

    # Check if overlay between FA and atlas is good
    # regtools.overlay_slices(fa, atlas_warped, None, 0, "FA", "warped atlas")
    # plt.show()  # Should be good!

    # Mask regions and store their mean values first region
    region_fa_means = []
    region_md_means = []
    for val in range(0, int(parcel_vals[-1])):
        region = atlas_warped == val
        region_fa_means.append(np.nanmean(fa[region]))
        region_md_means.append(np.nanmean(md[region]))

    # Save Regional FA means
    region_fa_means_df = pd.DataFrame()
    region_fa_means_df["Parcel"] = parcellation_names
    region_fa_means_df["FA_mean"] = region_fa_means
    region_fa_means_df.to_csv(srcp + f"/{subname}_regional_fa_means.csv", sep=",", index=False)

    # Save Regional MD means
    region_md_means_df = pd.DataFrame()
    region_md_means_df["Parcel"] = parcellation_names
    region_md_means_df["MD_mean"] = region_md_means
    region_md_means_df.to_csv(srcp + f"/{subname}_regional_md_means.csv", sep=",", index=False)

    # If we want regions in pairs (joined LR // region 0 is just background!):
    region_pair_fa_means = []
    region_pair_md_means = []
    for val in range(1, int(parcel_vals[-1]), 2):
        region = np.logical_or(atlas_warped == val, atlas_warped == val + 1)
        region_pair_fa_means.append(np.nanmean(fa[region]))
        region_pair_md_means.append(np.nanmean(md[region]))

    half_names = parcellation_names[range(1, parcellation_names.shape[0], 2)]
    dual_parcellation_names = [x[:-2] for x in half_names]
    # Save Regional FA means
    region_pair_fa_means_df = pd.DataFrame()
    region_pair_fa_means_df["Parcel"] = dual_parcellation_names
    region_pair_fa_means_df["FA_mean"] = region_pair_fa_means
    region_pair_fa_means_df.to_csv(srcp + f"/{subname}_region_pairs_fa_means.csv", sep=",", index=False)

    # Save Regional MD means
    region_pair_md_means_df = pd.DataFrame()
    region_pair_md_means_df["Parcel"] = dual_parcellation_names
    region_pair_md_means_df["MD_mean"] = region_pair_md_means
    region_pair_md_means_df.to_csv(srcp + f"/{subname}_region_pairs_md_means.csv", sep=",", index=False)

    print(f"Finished with subject {subname}")