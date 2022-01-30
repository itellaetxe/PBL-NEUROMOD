import numpy as np
from dipy.tracking import utils
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.io.utils import (create_nifti_header, get_reference_info,
                           is_header_compatible)
from dipy.io.image import load_nifti_data, load_nifti, save_nifti
import nibabel as nib
from matplotlib import pyplot as plt
from dipy.align.reslice import reslice
from dipy.align.imaffine import (MutualInformationMetric, AffineRegistration, transform_origins)
from dipy.align.transforms import (TranslationTransform3D, RigidTransform3D, AffineTransform3D)
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.metrics import CCMetric
from dipy.tracking.streamline import deform_streamlines
from dipy.viz import has_fury, window, actor
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_tractogram
from dipy.viz import regtools


kable = load_tractogram('Track0.trk', 'same')
affineT, dimensions, voxel_sizes, voxel_order = get_reference_info(kable)
B0_im, B0_affine = load_nifti('b0_resampled.nii.gz')
np.unique(B0_im.shape == dimensions)
atlas, atlas_affine = load_nifti('BN_Atlas_246_1mm.nii.gz')
std_im, std_affine, prev_voxel_size = load_nifti('HCP40_MNI_1.25mm.nii.gz', return_voxsize=True)
new_voxel_size = (1., 1., 1.)
std_im_rs, std_affine_rs = reslice(std_im, std_affine, prev_voxel_size, new_voxel_size)
print('ATLAS dimensions = ' + str(atlas.shape))
print('Resliced standard image dimensions = ' + str(std_im_rs.shape))
plt.subplot(121)
plt.imshow(std_im_rs[:,:, 120], cmap='gray')
plt.subplot(122)
plt.imshow(atlas[:,:, 120])
std_im = std_im_rs
std_affine = std_affine_rs # Overwrite the resliced volumes.
print('There is a little offset in the dimensions. (Why?)')

static = std_im
static_affine = std_affine
moving = B0_im
moving_affine = B0_affine
affine_map = transform_origins(static, static_affine, moving, moving_affine)
nbins = 32
sampling_prop = None
metric = MutualInformationMetric(nbins, sampling_prop)
level_iters = [10, 10, 5]
sigmas = [3.0, 1.0, 0.0]
factors = [4, 2, 1]
affine_reg = AffineRegistration(metric=metric, level_iters=level_iters,
                                sigmas=sigmas, factors=factors)
transform = TranslationTransform3D()

params0 = None
translation = affine_reg.optimize(static, moving, transform, params0,
                                  static_affine, moving_affine)
transformed = translation.transform(moving)
transform = RigidTransform3D()

rigid_map = affine_reg.optimize(static, moving, transform, params0,
                                static_affine, moving_affine,
                                starting_affine=translation.affine)
transformed = rigid_map.transform(moving)
transform = AffineTransform3D()
affine_reg.level_iters = [20, 20, 5]
highres_map = affine_reg.optimize(static, moving, transform, params0,
                                  static_affine, moving_affine,
                                  starting_affine=rigid_map.affine)
transformed = highres_map.transform(moving)
metric = CCMetric(3)
level_iters = [10, 10, 5]
sdr = SymmetricDiffeomorphicRegistration(metric, level_iters)

mapping = sdr.optimize(static, moving, static_affine, moving_affine,
                       highres_map.affine)
warped_moving = mapping.transform(moving)
vox_size = 1
# Create an isocentered affine
target_isocenter = np.diag(np.array([-vox_size, vox_size, vox_size, 1]))
# Take the off-origin affine capturing the extent contrast between the mean B0 image and the template
origin_affine = affine_map.affine.copy()
mni_streamlines = deform_streamlines(
    kable.streamlines, deform_field=mapping.get_forward_field(),
    stream_to_current_grid=target_isocenter,
    current_grid_to_world=origin_affine, stream_to_ref_grid=target_isocenter,
    ref_grid_to_world=np.eye(4))
std_Nifti = nib.Nifti1Image(std_im, std_affine)
sft = StatefulTractogram(mni_streamlines, std_Nifti, Space.RASMM)
save_tractogram(sft, 'mni-lr-superiorfrontal.trk', bbox_valid_check=False)
tracks_affine, dims, voxel_szs, voxel_ords = get_reference_info(sft)
M, grouping = utils.connectivity_matrix(sft,tracks_affine, label_volume=atlas.astype(np.uint8), return_mapping=False, mapping_as_streamlines=False)



