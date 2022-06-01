import numpy as np
from dash.exceptions import PreventUpdate
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF
from scipy.spatial import ConvexHull

__all__ = ['gpr_based_background_single_spectrum']


def convex_hull(wavenumbers, spectrum):
    points = np.hstack( wave)



def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx


def gpr_based_background_single_spectrum(wavenumbers,
                                         spectrum,
                                         control_points,
                                         control_regions,
                                         mask,  # ????????
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
    if not control_points:
        return np.zeros_like(wavenumbers)
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
