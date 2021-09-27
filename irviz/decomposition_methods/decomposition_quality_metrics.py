import numpy as np
import einops


def spectral_correlation(data, reconstructed, mask, spectral_mask):
    """

    Parameters
    ----------
    data: IR data tensor (Nwav, Nx, Ny)
    reconstructed: Reconstructed data tensor (Nwav, Nx, Ny)
    mask: pixel use mask (Nx, Ny)
    spectral_mask: wavenumber use mask (Nwav)

    Returns: A (Nx Ny) map of correlation coefficients.
    -------

    """
    _, Nx, Ny = data.shape
    mask = einops.rearrange(mask, "Nx Ny -> (Nx Ny)")
    # spatial selection
    x = einops.rearrange(data, "Nwav Nx Ny -> (Nx Ny) Nwav")[mask.astype(bool), :]
    y = einops.rearrange(reconstructed, "Nwav Nx Ny -> (Nx Ny) Nwav")[mask.astype(bool), :]
    # spectral selection
    x = x[:, spectral_mask.astype(bool)]
    y = y[:, spectral_mask.astype(bool)]

    # compute correlation
    mx = np.mean(x, axis=1)
    my = np.mean(y, axis=1)
    sx = np.std(x, axis=1)
    sy = np.std(y,axis=1)
    sxy = np.mean(  (x.T-mx).T*(y.T-my).T, axis=1 )
    correlation = sxy/(sx*sy)

    # remap back to proper space
    result = np.zeros((Nx,Ny))
    result = einops.rearrange(result, "Nx Ny -> (Nx Ny)")
    result[~mask.astype(bool)] = None
    result[mask.astype(bool)] = correlation
    result = einops.rearrange( result, "(Nx Ny) -> Nx Ny", Nx=Nx, Ny=Ny)
    return result



