import warnings

import h5pickle as h5
import numpy as np
import sklearn.decomposition
from dask import array as da
from PIL import Image
import einops

from irviz import DecompositionTuner
from irviz.background_filters.emsc import emsc_background_single_spectrum
from irviz.background_filters.gpr import gpr_based_background_single_spectrum
from irviz.clustering_methods import agglomerative_clustering
from irviz.clustering_methods.kmeans_clustering import kmeansClustering
from irviz.decomposition_methods import kernel_PCA, simple_PCA
from irviz.displays.background_isolator import BackgroundIsolator

TEST_FILE = 'E:\\BP-area3a.h5'
# TEST_FILE = '/home/ihumphrey/Dev/irviz/data/ir_stxm.h5'
# TEST_FILE = '/home/ihumphrey/Dev/irviz/data/BP-area3a.h5'
OPTICAL_TEST_FILE = 'E:\\BP-area3a_clean.JPG'
# TEST_FILE = '/home/ihumphrey/Dev/irviz/data/BP-area3a.h5'


def process_all_points(wavenumbers, full_map, parameter_set):
    """
    Parameters
    ----------
    wavenumbers: wavenumbers vector
    full_map: full spectral cube data set
    parameter_set: list of BackgroundIsolator.parameter_set

    Returns
    -------

    """
    # compute mean and std of spectra
    mean_spectra = []
    std_spectra = []
    for pset in parameter_set:
        sel = np.array(pset['map_mask']).astype(bool)
        this_group = full_map[:, sel]
        this_mean = np.mean(this_group, axis=1)
        this_std = np.std(this_group, axis=1)
        mean_spectra.append(this_mean)
        std_spectra.append(this_std)

    # loop over all spectra and make a Z-score map
    z_scores = np.zeros((len(mean_spectra), full_map.shape[1], full_map.shape[2]))
    tmp = einops.rearrange(full_map, "N x y -> x y N")
    for ii in range(z_scores.shape[0]):
        mu = mean_spectra[ii]
        sig = std_spectra[ii]
        score = np.abs(np.mean((tmp - mu) / sig, axis=-1))
        z_scores[ii, :, :] = score
    groups = np.argmin(z_scores, axis=0)

    del z_scores, tmp
    # here are the groups

    # now we loop over all points and run the individual algorithms

    corrected_spectra = np.zeros_like(full_map)
    backgrounds = np.zeros_like(full_map)

    for ii in range(full_map.shape[1]):
        for jj in range(full_map.shape[2]):
            this_group = groups[ii, jj]
            this_spectrum = full_map[:, ii, jj]
            these_values = parameter_set[this_group]['values']
            anchor_points = anchor_regions = None
            # get anchor points
            if len(parameter_set[this_group]['anchor_points']) > 0:
                single_spectrum_function = gpr_based_background_single_spectrum  # assign background fitting func
                anchors = parameter_set[this_group]['anchor_points']

                for item in anchors:
                    anchor_points.append(item['x'])
                anchor_points = np.array(anchor_points)
            elif len(parameter_set[this_group]['anchor_regions']) > 0:  # get anchor regions
                single_spectrum_function = emsc_background_single_spectrum  # assign background fitting func
                anchor_regions = parameter_set[this_group]['anchor_regions']

            input_part1 = {'wavenumbers': wavenumbers,
                           'spectrum': this_spectrum,
                           'control_points': anchor_points,
                           'control_regions': anchor_regions,
                           'mask': None}
            background = single_spectrum_function(**input_part1, **these_values)
            corrected = this_spectrum - background
            corrected_spectra[:, ii, jj] = corrected
            backgrounds[:, ii, jj] = background

    return corrected_spectra, backgrounds, groups


def open_optical_file(jpg_file):
    return np.asarray(Image.open(jpg_file))


def open_map_file(h5_file):
    f = h5.File(h5_file, 'r')
    data = f[next(iter(f.keys()))]['data']['image']['image_cube']
    wavenumbers = f[next(iter(f.keys()))]['data']['wavenumbers'][:]
    xy = f[next(iter(f.keys()))]['data']['xy'][:]
    bounds = [wavenumbers,
              (xy.T[1].min(), xy.T[1].max()),
              (xy.T[0].min(), xy.T[0].max())]
    return da.from_array(data).transpose(2, 0, 1), bounds


def open_ir_file(h5_file):
    f = h5.File(h5_file, 'r')
    data = f['irmap']['DATA']['data']
    bounds_grid = f['irmap']['DATA']['energy'][:], f['irmap']['DATA']['sample_y'][:], f['irmap']['DATA']['sample_x'][:]
    bounds = list(map(lambda grid: (grid.min(), grid.max()), bounds_grid))
    print(bounds)
    return da.from_array(data), bounds


if __name__ == "__main__":    # data, bounds = open_ir_file(TEST_FILE)
    data, bounds = open_map_file(TEST_FILE)
    optical = np.flipud(open_optical_file(OPTICAL_TEST_FILE))
    model = sklearn.decomposition.PCA(n_components=3)

    with warnings.catch_warnings():
        # Ignore future warning generated by dask (at worst, we do .compute beforehand, which probably already happens)
        warnings.simplefilter('ignore', FutureWarning)
        reshaped_data = data.transpose(1, 2, 0).reshape(-1, data.shape[0])
        decomposition = model.fit_transform(reshaped_data).T.reshape(-1, *data.shape[1:])

    cluster_labels = np.argmax(decomposition, axis=0)
    cluster_label_names = ['Alpha', 'Bravo', 'Charlie']

    static_mask = np.random.random(data.shape[1:]) < .01  # randomly mask 1% of pixels

    viewer = DecompositionTuner(data=data,
                                bounds=bounds,
                                static_mask=static_mask,
                                parameter_sets=[{'name': name,
                                                 'map_mask': cluster_labels == i} for i, name in enumerate(cluster_label_names)],
                                decomposition_functions={'Kernel PCA': kernel_PCA,
                                                         'Simple PCA': simple_PCA},
                                cluster_functions={'Agglomerative Clustering': agglomerative_clustering,
                                                   'K-means Clustering': kmeansClustering})

    viewer.run_server(run_kwargs=dict(debug=True, dev_tools_ui=True))
