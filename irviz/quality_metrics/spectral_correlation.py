import numpy as np
import einops

__all__ = ['spectral_correlation']

# TODO: change spectral mask to accept list of dicts representing masked regions
from irviz.utils.mapper import selection_brackets_to_bool_array


def spectral_correlation(wavenumbers, data, eigen_spectra, components, mask, spectral_mask):
    """

    Parameters
    ----------
    wavenumbers: spectral coordinates (Nwave,)
    data: IR data tensor (Nwav, Nx, Ny)
    eigen_spectra: Decomposition spectral eigenvectors (Ncomp, Nwav)
    components: Decomposition component amplitudes (Ncomp, Nx, Ny)
    mask: pixel use mask (Nx, Ny)
    spectral_mask: wavenumber use mask (Nwav)

    Returns: A (Nx Ny) map of correlation coefficients.
    -------

    """
    _, Nx, Ny = data.shape
    mask = einops.rearrange(mask, "Nx Ny -> (Nx Ny)")
    # spatial selection
    x = einops.rearrange(data, "Nwav Nx Ny -> (Nx Ny) Nwav")[mask.astype(bool), :]
    components = einops.rearrange(components, 'Ncomp Nx Ny -> (Nx Ny) Ncomp')[mask.astype(bool), :]
    # spectral selection
    spectral_mask = selection_brackets_to_bool_array(spectral_mask, wavenumbers)
    x = x[:, spectral_mask.astype(bool)]
    eigen_spectra = eigen_spectra[:, spectral_mask.astype(bool)]

    # compute correlation
    mx = np.mean(x, axis=1)
    my = np.zeros(mx.shape) # spectral mean
    sx = np.std(x, axis=1)
    sy = np.zeros(sx.shape)
    sxy = np.zeros(sy.shape)

    for i in range(x.shape[0]):  # could numba-ize this
        y = np.dot(eigen_spectra.T, components[i])  # y is a reconstructed single spatial pixel slice, comparable to x[i]
        my[i] = np.mean(y)
        sy[i] = np.std(y)
        sxy[i] = np.mean((x[i]-mx[i])*(y-my[i]))

    correlation = sxy/(sx*sy)

    # remap back to proper space
    result = np.zeros((Nx,Ny))
    result = einops.rearrange(result, "Nx Ny -> (Nx Ny)")
    result[~mask.astype(bool)] = None
    result[mask.astype(bool)] = correlation
    result = einops.rearrange( result, "(Nx Ny) -> Nx Ny", Nx=Nx, Ny=Ny)
    return result
