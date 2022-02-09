from sklearn.decomposition import KernelPCA as skKernelPCA
import numpy as np
from irviz.utils.mapper import einops_data_mapper, selection_brackets_to_bool_array
from irviz.quality_metrics import spectral_correlation_from_map

def kernel_PCA(wavenumbers,
               spectral_map,
               pixel_usage_mask,
               spectral_mask,
               n_components=0,
               kernel="rbf",
               gamma=0,
               alpha=1e-4,
               ):
    """

    Parameters
    ----------
    wavenumbers : the wavenumbers, an array of size (Nwav)
    spectral_map : an (Nwav,Nx,Ny) array of spectral data
    pixel_usage_mask: a boolean map of size (Nx,Ny) with elements set to True for pixels of interest
    spectral_mask: A boolean array of size (Nwav) for spectral regions of interest
    n_components: The number of components we want to extract; if set to 0, it is determined automatically.
    kernel: The kernel type we want to use.
    gamma: Control poarameter gamma, set to zero for default behavior.
    alpha: Control parameter alpha. This value needs tuning if you want the reconstructed data to be meaningful.

    Returns:
        U: Low dimensional representation of the spectra in n_components - this is a (n_component,Nx,Ny) array
        V: Kernel PCA doesn't give basis vectors like normal PCA, so we return np.empty([])
        Recon: The reconstructed data from the kernelPCA object
        quality: (Nx, Ny) map describing the decomposition quality
    -------

    """

    # engineer back in default behavior
    if n_components == 0:
        n_components = None
    if gamma == 0:
        gamma = None

    # check kernel type
    assert kernel in ["linear", "poly", "rbf", "sigmoid", "cosine"]

    shape = spectral_map.shape
    if pixel_usage_mask is None:
        pixel_usage_mask = np.ones(shape[1:]).astype(bool)
    if spectral_mask is None:
        spectral_mask = np.ones((shape[0])).astype(bool)
    else:
        spectral_mask = selection_brackets_to_bool_array(spectral_mask, wavenumbers)

    data_mapper_object = einops_data_mapper(spectral_map.shape, pixel_usage_mask, spectral_mask)
    data = data_mapper_object.spectral_tensor_to_spectral_matrix(np.asarray(spectral_map))

    # now we are ready for decomposition
    transformer = skKernelPCA(n_components=n_components,
                            kernel=kernel,
                            gamma=gamma,
                            alpha=alpha,
                            fit_inverse_transform=True)
    U = transformer.fit_transform(data)
    Recon = transformer.inverse_transform(U)

    U_out = data_mapper_object.matrix_to_tensor(U)
    Recon_out = data_mapper_object.spectral_matrix_to_spectral_tensor(Recon)
    Q = spectral_correlation_from_map.spectral_correlation_from_map(wavenumbers, spectral_map, Recon_out, pixel_usage_mask, spectral_mask)
    V = np.empty([])
    return U_out, V, Q  # Recon_out TODO: Ask Peter if this is necessary









