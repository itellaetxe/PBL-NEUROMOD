import os
import re
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import scipy.io

from dipy.io.streamline import load_tractogram
from dipy.io.utils import get_reference_info
from dipy.tracking import utils
from dipy.io.image import load_nifti_data, load_nifti, save_nifti
from dipy.align.reslice import reslice
from dipy.viz import regtools
from dipy.align.imaffine import (AffineMap, MutualInformationMetric, AffineRegistration)
from dipy.align.transforms import (TranslationTransform3D, RigidTransform3D, AffineTransform3D)
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.imwarp import DiffeomorphicMap
from dipy.align.metrics import CCMetric


def diffeomorphic_reg(sdr, template_img, moving_img, template_affine, moving_affine, affine):
    print("--> Diffeomorphic registration started.")

    diffeom_mapping = sdr.optimize(template_img, moving_img, template_affine, moving_affine, affine.affine)

    diffeom_transformed = diffeom_mapping.transform(moving_img)
    regtools.overlay_slices(template_img, diffeom_transformed, None, 0, "Template", "Transformed")
    print("--> Diffeomorphic registration finished.")

    return diffeom_transformed, diffeom_mapping


def affine_reg(affreg, template_img, moving_img, template_affine, moving_affine, rigid):
    print("--> Affine registration started.")

    transform_affine = AffineTransform3D()
    params0 = None
    affreg.level_iters = [1000, 1000, 100]
    affine = affreg.optimize(template_img, moving_img, transform_affine, params0, template_affine, moving_affine, starting_affine=rigid.affine)

    affine_transformed = affine.transform(moving_img)
    regtools.overlay_slices(template_img, affine_transformed, None, 0, "Template", "Transformed")
    print("--> Affine registration finished.")

    return affine_transformed, affine


def rigid_reg(affreg, template_img, moving_img, template_affine, moving_affine, translation):
    print("--> Rigid registration started.")
    transform_rigid = RigidTransform3D()
    params0 = None
    rigid = affreg.optimize(template_img, moving_img, transform_rigid, params0, template_affine, moving_affine, starting_affine=translation.affine)

    rigidbody_transformed = rigid.transform(moving_img)
    regtools.overlay_slices(template_img, rigidbody_transformed, None, 0, "Template", "Transformed")
    print("--> Rigid registration finished.")

    return rigidbody_transformed, rigid


def translation_reg(affreg, template_img, moving_img, template_affine, moving_affine):
    print("--> Translation registration started.")
    transform_translation = TranslationTransform3D()
    params0 = None

    # This starts computing the translation
    translation = affreg.optimize(template_img, moving_img, transform_translation, params0, template_affine, moving_affine)

    translated_img = translation.transform(moving_img)
    regtools.overlay_slices(template_img, translated_img, None, 0, "Template", "Transformed")
    print("--> Translation registration finished.")

    return translated_img, translation


def reslice_standard(template_img, template_affine, voxel_size):
    new_voxel_size = (1., 1., 1.)

    if voxel_size != new_voxel_size:
        print("--> STD is being resliced ...")
        template_img, template_affine = reslice(template_img, template_affine, voxel_size, new_voxel_size)

    return template_img, template_affine


# base_path = "D:/PROCESSED_ADNI_AD_GROUP/PROCESSED_AD_GROUP/"
base_path = "D:/PROCESSED_ADNI_CONTROL_GROUP/results/"
dirs = os.listdir(base_path)
r = re.compile("sub-")
filtered = [folder for folder in dirs if r.match(folder)]

# filtered = ['sub-ADNI002S6030_ses-M00', 'sub-ADNI003S4644_ses-M84', 'sub-ADNI014S6502_ses-M00']
for i in range(14, len(filtered), 1):
    print(f"_________________________{filtered[i]} started_________________________")
    # Load T1 (DWI) data
    print("--> Loading warped T1 ...")
    T1_path = base_path + filtered[i] + "/Register_T1"
    dirs = os.listdir(T1_path)
    t1_dirs = [folder for folder in dirs if "t1_warped" in folder]
    T1_path = T1_path + "/" + t1_dirs[0]
    moving_img, moving_affine = load_nifti(T1_path)

    # Load standard reference
    print("--> Loading STD ...")
    template_img, template_affine, voxel_size = load_nifti(base_path + 'HCP40_MNI_1mm.nii.gz', return_voxsize=True)

    # Checking the slice thicknes off the std. If it is not 1 px, then reslice
    template_img, template_affine = reslice_standard(template_img, template_affine, voxel_size)

    # Registration is now started by resampling both the std and the T1 to the same resolution
    identity = np.eye(4)
    affine_map = AffineMap(identity, template_img.shape, template_affine, moving_img.shape, moving_affine)
    resampled = affine_map.transform(moving_img)
    regtools.overlay_slices(template_img, resampled, None, 0, "Template", "Moving")

    # Obtention of the error between spaces (Mismatch metric) & definition of optimization strategy
    nbins = 32
    sampling_group = None
    metric = MutualInformationMetric(nbins, sampling_group)

    level_iters = [10, 10, 5]
    sigmas = [3.0, 1.0, 0.0]
    factors = [4, 2, 1]
    affreg = AffineRegistration(metric=metric, level_iters=level_iters, sigmas=sigmas, factors=factors)

    translated_img, translation = translation_reg(affreg, template_img, moving_img, template_affine, moving_affine)
    rigidbody_transformed, rigid = rigid_reg(affreg, template_img, moving_img, template_affine, moving_affine, translation)
    affine_transformed, affine = affine_reg(affreg, template_img, moving_img, template_affine, moving_affine, rigid)

    # Now defining the error metric and the optimization strategy for the dipheomorphic registration
    metric = CCMetric(3)
    level_iters = [10, 10, 5]
    sdr = SymmetricDiffeomorphicRegistration(metric, level_iters)

    # Perform diffeomorphic registration
    diffeom_transformed, diffeom_mapping = diffeomorphic_reg(sdr, template_img, moving_img, template_affine, moving_affine, affine)

    # Apply the INVERSE DIFFEOMORPHIC MAPPING to the STANDARD image to see if it matches the moving image
    STD_inv_diffeom_transformed = diffeom_mapping.transform_inverse(template_img, 'linear')
    type(STD_inv_diffeom_transformed)
    regtools.overlay_slices(moving_img, STD_inv_diffeom_transformed, None, 0, "Template", "Moving") # In this case the moving image is "warped_inv"

    """
    Now that our registration is good, we need to apply this procedure to the atlas. For this:
    
    1.- Apply the inverse diffeomorphic transformation to the ATLAS.
    
    2.- Apply the INVERSE affine transformation to the ATLAS (including the rotation+translation that involves rigid body transformation)
    """

    # Load the atlas
    print("--> Loading atlas ...")
    atlas_img, atlas_affine = load_nifti(base_path + 'BN_Atlas_246_1mm.nii.gz')
    inv_atlas = diffeom_mapping.transform_inverse(atlas_img, 'linear')
    regtools.overlay_slices(moving_img, inv_atlas, None, 0, "Template", "Moving")

    # So now we see that the ATLAS is moved to the DWI space. The diffeomorphic map encapsulates
    # the previously done transformations as it was optimized with them being a starting point.
    print("--> Loading B0 ...")
    B0_path = base_path + filtered[i] + "/Resample_B0"
    dirs = os.listdir(B0_path)
    b0_dirs = [folder for folder in dirs if "b0_resampled" in folder]
    B0_path = B0_path + "/" + b0_dirs[0]
    #moving_img, moving_affine = load_nifti(B0_path)
    B0_im, B0_affine = load_nifti(B0_path)
    regtools.overlay_slices(B0_im, inv_atlas, None, 0, "B0", "Transformed Atlas")
    inv_atlas_affine = B0_affine
    # save_nifti('inverted_atlas.nii.gz', inv_atlas, inv_atlas_affine)

    print("--> Loading tractogram ...")
    trk_path = base_path + filtered[i] + "/PFT_Tracking"
    dirs = os.listdir(trk_path)[0]
    trk_path = trk_path + "/" + dirs
    #moving_img, moving_affine = load_nifti(trk_path)
    tractogram = load_tractogram(trk_path, 'same')
    tracks_affine, dims, voxel_szs, voxel_ords = get_reference_info(tractogram)
    print(tracks_affine)

    M, grouping = utils.connectivity_matrix(tractogram.streamlines,tracks_affine, label_volume=atlas_img.astype(np.uint8), return_mapping=True, mapping_as_streamlines=True)
    if not os.path.exists(base_path + filtered[i] + "/ConnMat/"):
        os.mkdir(base_path + filtered[i] + "/ConnMat/")
        print("--> ConnMat folder created ...")

    file_name = base_path + filtered[i] + "/ConnMat/" + "ConnMat_" + filtered[i]
    scipy.io.savemat(file_name + ".mat", {'myConnMat': M}, appendmat=True)
    np.savetxt(base_path + filtered[i] + "/ConnMat/" + "ConnMat_" + filtered[i] + ".csv", M, delimiter=",")
    print("--> Connectivity Matrix correctly stored.")
    # plt.figure()
    # plt.imshow(np.log1p(M), interpolation='nearest')
    # plt.show()
    # plt.savefig(filtered[i] + "ConnMat/" + 'ConnMat.png')
    # plt.close('all')

