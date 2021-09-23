import pacmap
import numpy as np
from irviz.utils.mapper import einops_data_mapper

def pacmap_decompose(wavenumbers,
              spectral_map,
              pixel_usage_mask,
              spectral_mask,
              n_dims=2):

    # engineer in default behavior
    assert n_dims > 0
    shape = spectral_map.shape
    if pixel_usage_mask is None:
        pixel_usage_mask = np.ones(shape[1:]).astype(bool)
    if spectral_mask is None:
        spectral_mask = np.ones((shape[0])).astype(bool)
    data_mapper_object = einops_data_mapper(spectral_map.shape, pixel_usage_mask, spectral_mask)
    data = data_mapper_object.spectral_tensor_to_spectral_matrix(spectral_map)
    embedder = pacmap.PaCMAP(n_dims=n_dims)
    U = embedder.fit_transform(data)
    U_out = data_mapper_object.matrix_to_tensor(U)
    return U_out, None, None

