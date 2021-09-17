from sklearn.decomposition import KernelPCA as skKernelPCA
import einops
import numpy as np

def KernelPCA(wavenumbers,
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
    spectral_map : an (Nx,Ny,Nwav) array of spectral data
    pixel_usage_mask: a boolean map of size (Nx,Ny) with elements set to True for pixels of interest
    spectral_mask: A boolean array of size (Nwav) for spectral regions of interest
    n_components: The number of components we want to extract; if set to 0, it is determined automatically.
    kernel: The kernel type we want to use.
    gamma: Control poarameter gamma, set to zero for default behavior.
    alpha: Control parameter alpha. This value needs tuning if you want the reconstructed data to be meaningful.

    Returns:
        U: Low dimensional representation of the spectra in n_components - this is a (Nx,Ny,n_component) array
        V: Kernel PCA doesn't give basis vectors like normal PCA, so we return None
        Recon: The reconstructed data from the kernelPCA object
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


    # flatten data
    data = einops.rearrange(spectral_map, " Nwav Nx Ny -> (Nx Ny) Nwav")
    # select wavenumbers
    data = data[:, spectral_mask.astype(bool)]

    # select pixels
    mask = einops.rearrange(pixel_usage_mask, " Nx Ny -> (Nx Ny)")
    data = data[mask.astype(bool)] # make bool if int map

    # now we are ready for decomposition
    transformer = skKernelPCA(n_components=n_components,
                            kernel=kernel,
                            gamma=gamma,
                            alpha=alpha,
                            fit_inverse_transform=True)
    U = transformer.fit_transform(data)
    Recon = transformer.inverse_transform(U)

    # map things back into shape

    # First U
    U_out = np.zeros( (shape[1]*shape[2], U.shape[-1]) )
    U_out[ ~mask.astype(bool), : ] = None
    U_out[ mask.astype(bool),:] = U
    U_out = einops.rearrange(U_out, "(Nx Ny) C -> C Nx Ny", Nx=shape[1], Ny=shape[2] )

    # now build a reconstructed dataset
    Recon_out = np.zeros(shape)
    Fill_mask = np.ones(shape)

    Recon_out = einops.rearrange(Recon_out, "Nwav Nx Ny -> (Nx Ny) Nwav")
    Fill_mask = einops.rearrange(Fill_mask, "Nwav Nx Ny -> (Nx Ny) Nwav")
    Fill_mask[ ~mask.astype(bool), :] = 0
    Fill_mask[:, ~spectral_mask.astype(bool)] = 0
    Fill_mask = einops.rearrange(Fill_mask, " A C -> (A C) ")

    # set non-used pixels to zero
    Recon_out[ : , : ] = None
    Recon = einops.rearrange(Recon, "A C -> (A C)")
    Recon_out = einops.rearrange(Recon_out, "A C -> (A C)")
    Recon_out[Fill_mask.astype(bool)] = Recon
    #reshape to proper output shape
    Recon_out = einops.rearrange(Recon_out, "(Nx Ny Nwav) -> Nwav Nx Ny", Nx=shape[1], Ny=shape[2], Nwav=shape[0])

    V = None
    return U_out, V, Recon_out















