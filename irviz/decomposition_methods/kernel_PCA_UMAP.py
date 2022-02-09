from sklearn.decomposition import KernelPCA as skKernelPCA
import numpy as np
from irviz.utils.mapper import einops_data_mapper, selection_brackets_to_bool_array
from irviz.decomposition_methods import kernel_PCA
import einops
from irviz.quality_metrics import spectral_correlation_from_map
import umap


def kernel_PCA_UMAP(wavenumbers,
                    spectral_map,
                    pixel_usage_mask,
                    spectral_mask,
                    kernelComponents=5,
                    kernel="rbf",
                    kernelGamma=0,
                    kernelAlpha=1e-3,
                    umapDensLambda=5.0,
                    umapRandomState=42,
                    umapUseFraction=1.0
               ):
    """

    Parameters
    ----------
    wavenumbers : the wavenumbers, an array of size (Nwav)
    spectral_map : an (Nwav,Nx,Ny) array of spectral data
    pixel_usage_mask : a boolean map of size (Nx,Ny) with elements set to True for pixels of interest
    spectral_mask : A selection of which wavenumbers to use
    kernel_components : The number of components use in kernelPCA.
    kernel : the kernel for kernel PCA ; rbf is fine
    kernel_gamma : the gamme parameter. default 0 is ok
    kernel_alpha : might need tuning for optimal reconstruction. default of 1e-3 should be ok.
    umap_neighbours : The number of neighbours used in UMAP
    umap_min_dist : the minimum distance in umap
    umap_dens_lambda : the densmap lambda parameter to use in UMAP
    umap_random_state : random state variable to make things reproducable
    umap_use_fraction : determines how much data is used for umap, speeds things up if needed

    Returns
    -------
        U: Low dimensional representation of the spectra in n_components - this is a (n_component,Nx,Ny) array
        V: Kernel PCA doesn't give basis vectors like normal PCA, so we return np.empty([])
        Recon: The reconstructed data from the kernelPCA object
        quality: (Nx, Ny) map describing the decomposition quality
    """

    # first we run kernelPCA
    U, _V, Q = kernel_PCA(wavenumbers=wavenumbers,
                         spectral_map=spectral_map,
                         pixel_usage_mask=pixel_usage_mask,
                         spectral_mask=spectral_mask,
                         n_components=kernelComponents,
                         kernel=kernel,
                         gamma=kernelGamma,
                         alpha=kernelAlpha)
    print(U.shape, "<-----")
    # now that we have a kernel PCA embedding U, we run a Umap on it.

    pseudo_mask = np.ones(kernelComponents)
    mapping_object = einops_data_mapper(U.shape, pixel_usage_mask, pseudo_mask)

    tmp_U = mapping_object.spectral_tensor_to_spectral_matrix(U)

    # get a subset of the data
    rng = np.random.default_rng(umapRandomState)
    N = tmp_U.shape[0]
    tmp_rand = rng.random(N)
    selection = tmp_rand < umapUseFraction
    data_for_umap = tmp_U[selection,:]

    # now build a UMAP reducer
    reducer = umap.UMAP(
        densmap=True,
        dens_lambda=umapDensLambda,
        random_state=umapRandomState,
        verbose=False).fit(data_for_umap)

    transformed_data = reducer.transform(tmp_U)

    # map it back to where we need it
    umap_U = mapping_object.matrix_to_tensor(transformed_data)
    return umap_U, _V, Q














