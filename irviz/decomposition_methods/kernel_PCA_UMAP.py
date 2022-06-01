import numpy as np
from scipy.signal import savgol_filter
from scipy.sparse.linalg import svds
from irviz.utils.mapper import einops_data_mapper, selection_brackets_to_bool_array, multiset_mapper
from irviz.decomposition_methods.kernel_PCA import svd_map
import umap
from sklearn.decomposition import KernelPCA as skKernelPCA




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
    kernelGamma : the gamme parameter. default 0 is ok
    kernelAlpha : might need tuning for optimal reconstruction. default of 1e-3 should be ok.
    umapDensLambda : the densmap lambda parameter to use in UMAP
    umapRandomState : random state variable to make things reproducable
    umapUseFraction : determines how much data is used for umap, speeds things up if needed

    Returns
    -------
        U: Low dimensional representation of the spectra in n_components - this is a (n_component,Nx,Ny) array
        V: Kernel PCA doesn't give basis vectors like normal PCA, so we return np.empty([])
        Recon: The reconstructed data from the kernelPCA object
        quality: (Nx, Ny) map describing the decomposition quality
    """

    # first we run kernelPCA
    """U, _V, Q = kernel_PCA(wavenumbers=wavenumbers,
                         spectral_map=spectral_map,
                         pixel_usage_mask=pixel_usage_mask,
                         spectral_mask=spectral_mask,
                         n_components=kernelComponents,
                         kernel=kernel,
                         gamma=kernelGamma,
                         alpha=kernelAlpha)"""

    U, _V, Q = svd_map(wavenumbers=wavenumbers,
                         spectral_map=spectral_map,
                         pixel_usage_mask=pixel_usage_mask,
                         spectral_mask=spectral_mask,
                         n_components=kernelComponents)
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
        #densmap=True,
        #dens_lambda=umapDensLambda,
        random_state=umapRandomState,
        verbose=False).fit(data_for_umap)

    transformed_data = reducer.transform(tmp_U)

    # map it back to where we need it
    umap_U = mapping_object.matrix_to_tensor(transformed_data)
    return umap_U, _V, Q


def derivative_kernelPCA_UMAP(wavenumbers,
                                  spectral_map,
                                  pixel_usage_mask,
                                  spectral_mask,
                                  kernelComponents=6,
                                  #kernel="rbf",
                                  #kernelGamma=0,
                                  #kernelAlpha=1e-3,
                                  umapDensLambda=3.0,
                                  umapRandomState=42,
                                  #umapUseFraction=1.0,
                                  windowLength=11,
                                  polyOrder=5,
                                  derivativeOrder=1):
    """

    Parameters
    ----------
    wavenumbers :
    spectral_map :
    pixel_usage_mask :
    spectral_mask :
    kernelComponents :
    kernel :
    kernelGamma :
    kernelAlpha :
    umapDensLambda :
    umapRandomState :
    umapUseFraction :
    windowLength :
    polyOrder :
    derivativeOrder :

    Returns
    -------

    """

    print(pixel_usage_mask.shape, np.sum(pixel_usage_mask))


    kernel = "rbf"
    kernelGamma = 0
    kernelAlpha = 1e-3
    #umapDensLambda = 5.0,
    #umapRandomState = 42,
    umapUseFraction = 1.0
    windowLength = 7
    polyOrder = 3
    #derivativeOrder = 1

    if polyOrder > windowLength:
        windowLength = windowLength + polyOrder
    if derivativeOrder > polyOrder:
        polyOrder = polyOrder + derivativeOrder

    snd_der_map = savgol_filter(spectral_map,
                                window_length=windowLength,
                                polyorder=polyOrder,
                                deriv=derivativeOrder,
                                delta=1.0,
                                axis= 0 ,
                                mode='interp', cval=0.0)

    umap_U, _V, Q = kernel_PCA_UMAP(wavenumbers,
                                    snd_der_map,
                                    pixel_usage_mask,
                                    spectral_mask,
                                    kernelComponents=kernelComponents,
                                    kernel=kernel,
                                    kernelGamma=kernelGamma,
                                    kernelAlpha=kernelAlpha,
                                    umapDensLambda=umapDensLambda,
                                    umapRandomState=umapRandomState,
                                    umapUseFraction=umapUseFraction)
    return umap_U, _V, Q


def multimap_SVD_UMAP(wavenumbers,
                            spectral_maps,
                            pixel_usage_masks,
                            spectral_mask,
                            kernelComponents=10,
                            umapDensLambda=1.0,
                            umapRandomState=42,
                            windowLength=7,
                            polyOrder=3,
                            derivativeOrder=1):
    kernel = "rbf"
    gamma = 0.0
    alpha = 0.001
    umapUseFraction = 0.5

    # first we need to get derivatives
    these_maps = []
    shapes = []
    for this_map in spectral_maps:
        print("derivative")
        this_derivative = savgol_filter(this_map,
                                          window_length=windowLength,
                                          polyorder=polyOrder,
                                          deriv=derivativeOrder,
                                          delta=1.0,
                                          axis=0,
                                          mode='interp', cval=0.0)
        these_maps.append(this_derivative)
        shapes.append(this_derivative.shape)

    # build a multiset mapper object
    mapper_object = multiset_mapper(shapes, pixel_usage_masks, spectral_mask)

    # get a spectral matrix
    spectral_matrix = mapper_object.spectral_tensor_to_spectral_matrix(these_maps)
    print(spectral_matrix.shape)


    # now we are ready for decomposition
    #transformer = skKernelPCA(n_components=kernelComponents,
    #                        kernel=kernel,
    #                        gamma=gamma,
    #                        alpha=alpha,
    #                        fit_inverse_transform=False)

    U,S,V = svds(spectral_matrix, k=kernelComponents)
    #U = transformer.fit_transform(spectral_matrix)
    print(U.shape)
    # now build a UMAP reducer
    print("umap1")
    reducer = umap.UMAP(
        #densmap=True,
        #dens_lambda=umapDensLambda,

        random_state=umapRandomState,
        verbose=False).fit(U)
    print("umap2")
    transformed_data = reducer.transform(U)
    print(transformed_data.shape)

    print("remap")
    # we now need to split the U's back into individual maps
    Umap_maps = mapper_object.matrix_to_tensor(transformed_data)

    U = mapper_object.matrix_to_tensor(U)

    return U, Umap_maps, reducer


