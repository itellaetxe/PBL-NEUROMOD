from dipy.io.streamline import load_tractogram
import numpy as np
import nibabel as nib
from dipy.io.image import load_nifti
from dipy.io.utils import get_reference_info
from dipy.tracking import utils
import matplotlib.pyplot as plt

tractogram = load_tractogram('Track0.trk', 'same')
tracks_affine, dims, voxel_szs, voxel_ords = get_reference_info(tractogram)
print(tracks_affine)
dims

atlas_img, atlas_affine = load_nifti('inverted_atlas.nii.gz')
print(atlas_affine)
atlas_img.shape

M, grouping = utils.connectivity_matrix(tractogram.streamlines, tracks_affine,
                                        label_volume=atlas_img.astype(np.uint8),
                                        return_mapping=True, mapping_as_streamlines=True)
plt.imshow(np.log1p(M), interpolation='nearest', cmap='seismic')
plt.show()
