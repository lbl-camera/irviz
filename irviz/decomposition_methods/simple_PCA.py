import numpy as np
import dask.array as da
from sklearn.decomposition import PCA
from irviz.background_app import open_map_file
from irviz.background_filters.gpr import find_nearest

TEST_FILE = 'E:\\BP-area3a.h5'


def select_regions(wavenumbers, data_map, control_regions):
    """
    select sub-regions of wavenumbers from data_map and concatenate into a truncated data_map

    Parameters
    ----------
    wavenumbers: sample positions of the spectral dimension, ranges in [600, 4000]
    data_map: 3D spectral map of shape [Ny, Nx, Nw]
    control_regions: selected wavenumber regions of interest, list of dictionaries

    Returns
    -------
    Re-assembled data cube of shape [Ny, Nx, len(wav_regions)]
    """
    # parse wavenumber ROI
    w_regions = []
    for region in control_regions:
        wav1 = region['region_min']
        wav2 = region['region_max']
        if wav1 == None: wav1 = wavenumbers[0]
        if wav2 == None: wav2 = wavenumbers[-1]
        w_regions.append((wav1, wav2))
    # sort w_regions by region_min
    w_regions = sorted(w_regions, key=lambda x: x[0])
    idx = []
    for wav1, wav2 in w_regions:
        idx.append((find_nearest(wavenumbers, wav1), find_nearest(wavenumbers, wav2)))

    data_cube = []
    for ind1, ind2 in idx:
        data_cube.append(data_map[:, :, ind1:ind2])
    data_cube = np.concatenate(data_cube, axis=2)

    return data_cube

def masked_to_map(mask, data):
    """
    mapping after-mask 1D data or 2D data of shape [n_sample, n_features] back to full 3D spectral map

    Parameters
    ----------
    mask: binary 2D mask of shape [Ny, Nx], numpy array
    data: 1D data or 2D array of shape [n_samples, n_features], where n_samples = np.count_nonzero(mask.ravel())

    Returns
    -------
    data_cube: 3D array of shape [Ny, Nx, n_features], may contain np.NANs
    """
    Ny, Nx = mask.shape[0], mask.shape[1]
    if data.ndim == 1:
        data_cube = np.ones((Ny, Nx)) * np.NaN
    else:
        Nw = data.shape[1]
        data_cube = np.ones((Ny, Nx, Nw)) * np.NaN
    data_cube[mask] = data
    return data_cube


def simple_PCA(wavenumbers, data_map, mask, control_regions, n_components=5):
    """
    perform PCA decomposition of data_map

    Parameters
    ----------
    wavenumbers: sample positions of the spectral dimension, ranges in [600, 4000]
    data_map: 3D spectral map of shape [Nw, Ny, Nx], dask array
    mask: binary 2D mask of shape [Ny, Nx], numpy array
    control_regions: selected wavenumber regions of interest, list of dictionaries
    n_components: number of PCA components to retain

    Returns
    -------
    data_transform: PCA transformed spectral map of shape[Nw, Ny, Nx], may contain np.NANs, dask array
    pca.components_: PCA eigenvectors
    """
    data_map = np.array(data_map)
    data_map = data_map.transpose((1, 2, 0))
    data_cube = select_regions(wavenumbers, data_map, control_regions)
    data_list = data_cube[mask]
    pca = PCA(n_components=n_components)
    data_transform = pca.fit_transform(data_list)
    data_cube_transform = masked_to_map(mask, data_transform)
    data_cube_transform = da.from_array(data_cube_transform.transpose((2, 0, 1)))
    return data_cube_transform, pca.components_

def qscore_rms(wavenumbers, data_map, mask, control_regions, data_transform, components):
    """
    Compute quality scores of the reconstructed spectral map after decomposition

    Parameters
    ----------
    wavenumbers: sample positions of the spectral dimension, ranges in [600, 4000]
    data_map: 3D spectral map of shape [Nw, Ny, Nx], dask array
    mask: binary 2D mask of shape [Ny, Nx], numpy array
    control_regions: selected wavenumber regions of interest, list of dictionaries
    data_transform: PCA transformed spectral map of shape[Nw, Ny, Nx], dask array
    components: PCA eigenvectors

    Returns
    -------
    rms_map: quality score map, may contain np.NANs, numpy array
    """
    data_map, data_transform = np.array(data_map), np.array(data_transform)
    data_map, data_transform = data_map.transpose((1, 2, 0)), data_transform.transpose((1, 2, 0))
    data_cube = select_regions(wavenumbers, data_map, control_regions)
    data_list, data_transform = data_cube[mask], data_transform[mask]
    data_list_fit = data_transform @ components
    rms_err = np.linalg.norm(data_list - data_list_fit, axis=1)
    rms_err_map = masked_to_map(mask, rms_err)
    return rms_err_map

if __name__ == "__main__":
    data, bounds = open_map_file(TEST_FILE)
    wavenumbers = bounds[0]
    mask = np.random.random(data.shape[1:3]) > 0.5
    control_regions = [{'region_min': 1200, 'region_max': 1400}, {'region_min': 2700, 'region_max': 3000}]

    data_transform, vec = simplePCA(wavenumbers, data, mask, control_regions)
    assert data_transform.shape == (5, 29, 42), "shape of PCA transformed data is wrong."
    assert vec.shape.shape == (5, 260), "shape of PCA components are wrong."

    qs = qscore_rms(wavenumbers, data, mask, control_regions, data_transform, vec)
    assert qs.shape == (29, 42), "shape of quality score map is wrong"