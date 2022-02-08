import numpy as np
import einops

__all__ = ['spectral_correlation_from_map']


def spectral_correlation_from_map(wavenumbers, data, reconstruction, mask, spectral_mask):
    """

    Parameters
    ----------
    wavenumbers: spectral coordinates (Nwave,)
    data: IR data tensor (Nwav, Nx, Ny)
    reconstruction: IR data tensor of reconstructed map
    mask: pixel use mask (Nx, Ny)
    spectral_mask: wavenumber use mask (Nwav)

    Returns: A (Nx Ny) map of correlation coefficients.
    -------

    """
    _, Nx, Ny = data.shape
    mask = einops.rearrange(mask, "Nx Ny -> (Nx Ny)")
    # spatial selection
    x = einops.rearrange(data, "Nwav Nx Ny -> (Nx Ny) Nwav")[mask.astype(bool), :]
    y = einops.rearrange(reconstruction, 'Nrecon Nx Ny -> (Nx Ny) Nrecon')[mask.astype(bool), :]

    # spectral selection
    x = x[:, spectral_mask.astype(bool)]
    y = y[:, spectral_mask.astype(bool)]

    # compute correlation
    mx = np.mean(x, axis=1).reshape(-1,1)
    my = np.mean(y, axis=1).reshape(-1,1) # spectral mean
    sx = np.std(x, axis=1)
    sy = np.std(y, axis=1)
    sxy = np.mean( (x-mx)*(y-my), axis=1 )

    correlation = sxy/(sx*sy)

    # remap back to proper space
    result = np.zeros((Nx,Ny))
    result = einops.rearrange(result, "Nx Ny -> (Nx Ny)")
    result[~mask.astype(bool)] = None
    result[mask.astype(bool)] = correlation
    result = einops.rearrange( result, "(Nx Ny) -> Nx Ny", Nx=Nx, Ny=Ny)
    return result
