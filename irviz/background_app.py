import warnings
from functools import partial

import dash
import dash_bootstrap_components as dbc
import dash_html_components as html
import h5py as h5
import numpy as np
import sklearn.decomposition
import scipy.optimize
from scipy.signal import hilbert
from dask import array as da
from PIL import Image
import einops
from dash_bootstrap_templates import load_figure_template

import ryujin.utils.dash
from irviz.background_isolator import BackgroundIsolator
from irviz.viewer import Viewer



TEST_FILE = 'E:\\BP-area3a.h5'
# TEST_FILE = '/home/ihumphrey/Dev/irviz/data/ir_stxm.h5'
# TEST_FILE = '/home/ihumphrey/Dev/irviz/data/BP-area3a.h5'
OPTICAL_TEST_FILE = 'E:\\BP-area3a_clean.JPG'
# TEST_FILE = '/home/ihumphrey/Dev/irviz/data/BP-area3a.h5'

from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF, ConstantKernel as C


def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx


def GPR_based_background_single_spectrum(wavenumbers,
                                         spectrum,
                                         control_points,
                                         control_regions,
                                         mask, # ????????
                                         rbf_start=1000,
                                         rbf_low=500,
                                         rbf_high=1e8,
                                         C_start=1.0,
                                         C_low=1e-6,
                                         C_high=1e4
                                         ):
    """
    Build a background model using GPR

    :param wavenumbers: Input wavenumbers
    :param spectrum: input spectrum
    :param control_points: input control points, poicked manually
    :param rbf_kernel_params: kernel parameters, defaults ok
    :param constant_kernel_params: kernel parameters, defaults ok
    :return: a fitted background.
    """
    # gather the x values
    these_idxs = []
    for cp in control_points:
        these_idxs.append( find_nearest(wavenumbers, cp) )
    these_idxs = np.array(these_idxs)
    these_x = wavenumbers[these_idxs]
    these_y = spectrum[these_idxs]
    kernel = C(C_start,
               (C_low,
                C_high)) * \
             RBF(rbf_start, (rbf_low, rbf_high))

    gpr = gaussian_process.GaussianProcessRegressor(kernel=kernel).fit(these_x.reshape(-1,1),
                                                                       these_y.reshape(-1,1))
    tmp_bg = gpr.predict(wavenumbers.reshape(-1,1))
    return tmp_bg.flatten()

def apparent_spectrum_fit_function(wn, Z_ref, p, b, c, g):
    """
    Function used to fit the apparent spectrum
    :param wn: wavenumbers
    :param Z_ref: reference spectrum
    :param p: principal components of the extinction matrix
    :param b: Reference's linear factor
    :param c: Offset
    :param g: Extinction matrix's PCA scores (to be fitted)
    :return: fitting of the apparent specrum
    """
    A = b * Z_ref + c + np.dot(g, p)  # Extended multiplicative scattering correction formula
    return A

def find_nearest_number_index(array, value):
    """
    Find the nearest number in an array and return its index
    :param array:
    :param value: value to be found inside the array
    :return: position of the number closest to value in array
    """
    array = np.array(array)  # Convert to numpy array
    if np.shape(np.array(value)) == ():  # If only one value wants to be found:
        index = (np.abs(array - value)).argmin()  # Get the index of item closest to the value
    else:  # If value is a list:
        value = np.array(value)
        index = np.zeros(np.shape(value))
        k = 0
        # Find the indexes for all values in value
        for val in value:
            index[k] = (np.abs(array - val)).argmin()
            k += 1
        index = index.astype(int)  # Convert the indexes to integers
    return index


def Q_ext_kohler(wn, alpha):
    """
    Compute the scattering extinction values for a given alpha and a range of wavenumbers
    :param wn: array of wavenumbers
    :param alpha: scalar alpha
    :return: array of scattering extinctions calculated for alpha in the given wavenumbers
    """
    rho = alpha * wn
    Q = 2.0 - (4.0 / rho) * np.sin(rho) + (2.0 / rho) ** 2.0 * (1.0 - np.cos(rho))
    return Q

def Kohler_zero(wavenumbers, App, w_regions, alpha0=1.25, alpha1=49.95, n_alpha=150, n_components=8):
    """
    Correct scattered spectra using Kohler's algorithm
    :param wavenumbers: array of wavenumbers
    :param App: apparent spectrum
    :param m0: reference spectrum
    :param n_components: number of principal components to be calculated
    :return: corrected data
    """
    # Make copies of all input data:
    wn = np.copy(wavenumbers)
    A_app = np.copy(App)
    m_0 = np.zeros(len(wn))
    ii = np.argsort(wn)  # Sort the wavenumbers from smallest to largest
    # Sort all the input variables accordingly
    wn = wn[ii]
    A_app = A_app[ii]
    m_0 = m_0[ii]

    # Initialize the alpha parameter:
    alpha = np.linspace(alpha0, alpha1, n_alpha) * 1.0e-4  # alpha = 2 * pi * d * (n - 1) * wavenumber
    p0 = np.ones(2 + n_components)  # Initialize the initial guess for the fitting

    # # Initialize the extinction matrix:
    Q_ext = np.zeros((np.size(alpha), np.size(wn)))
    for i in range(np.size(alpha)):
        Q_ext[i][:] = Q_ext_kohler(wn, alpha=alpha[i])

    # Perform PCA of Q_ext:
    pca = sklearn.decomposition.IncrementalPCA(n_components=n_components)
    pca.fit(Q_ext)
    p_i = pca.components_  # Extract the principal components

    # print(np.sum(pca.explained_variance_ratio_)*100)  # Print th explained variance ratio in percentage
    w_indexes = []
    for pair in w_regions:
        min_pair = min(pair)
        max_pair = max(pair)
        ii1 = find_nearest_number_index(wn, min_pair)
        ii2 = find_nearest_number_index(wn, max_pair)
        w_indexes.extend(np.arange(ii1, ii2))
    wn_w = np.copy(wn[w_indexes])
    A_app_w = np.copy(A_app[w_indexes])
    m_w = np.copy(m_0[w_indexes])
    p_i_w = np.copy(p_i[:, w_indexes])

    def min_fun(x):
        """
        Function to be minimized by the fitting
        :param x: array containing the reference linear factor, the offset, and the PCA scores
        :return: function to be minimized
        """
        bb, cc, g = x[0], x[1], x[2:]
        # Return the squared norm of the difference between the apparent spectrum and the fit
        return np.linalg.norm(A_app_w - apparent_spectrum_fit_function(wn_w, m_w, p_i_w, bb, cc, g)) ** 2.0

    # Minimize the function using Powell method
    res = scipy.optimize.minimize(min_fun, p0, bounds=None, method='Powell')
    # print(res)  # Print the minimization result
    # assert(res.success) # Raise AssertionError if res.success == False

    b, c, g_i = res.x[0], res.x[1], res.x[2:]  # Obtain the fitted parameters

    # Apply the correction to the apparent spectrum
    Z_corr = (A_app - c - np.dot(g_i, p_i))  # Apply the correction
    base = np.dot(g_i, p_i)

    return Z_corr, base + c

def EMSC_background_single_spectrum(wavenumbers, spectrum, control_points, control_regions, mask,
                                    alpha0=1.25,
                                    alpha1=49.95,
                                    n_alpha=150,
                                    n_Qpca=8):
    """

    Parameters
    ----------
    wavenumbers
    spectrum
    control_points
    control_regions
    mask
    alpha0
    alpha1
    n_alpha
    n_Qpca

    Returns
    -------

    """
    w_regions = []
    for region in control_regions:
        wav1 = region['region_min']
        wav2 = region['region_max']
        if wav1 == None: wav1 = wavenumbers[0]
        if wav2 == None: wav2 = wavenumbers[-1]
        w_regions.append((wav1, wav2))

    _, baseline = Kohler_zero(wavenumbers, spectrum, w_regions,
                                      alpha0=alpha0, alpha1=alpha1, n_alpha=n_alpha, n_components=n_Qpca)

    return baseline


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
                single_spectrum_function = GPR_based_background_single_spectrum  # assign background fitting func
                anchors = parameter_set[this_group]['anchor_points']

                for item in anchors:
                    anchor_points.append(item['x'])
                anchor_points = np.array(anchor_points)
            elif len(parameter_set[this_group]['anchor_regions']) > 0:  # get anchor regions
                single_spectrum_function = EMSC_background_single_spectrum  # assign background fitting func
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
    bounds = [(wavenumbers.min(), wavenumbers.max()),
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

    viewer = BackgroundIsolator(data=data,
                                bounds=bounds,
                                parameter_sets=[{'name': name,
                                                 'map_mask': cluster_labels == i} for i, name in enumerate(cluster_label_names)],
                                background_function=GPR_based_background_single_spectrum)

    viewer.run_server(run_kwargs=dict(debug=True))
