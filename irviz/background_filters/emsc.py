import numpy as np
import scipy.optimize
import sklearn.decomposition

__all__ = ['emsc_background_single_spectrum']


def emsc_background_single_spectrum(wavenumbers, spectrum, control_points, control_regions, mask,
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

    _, baseline = kohler_zero(wavenumbers, spectrum, w_regions,
                              alpha0=alpha0, alpha1=alpha1, n_alpha=n_alpha, n_components=n_Qpca)

    return baseline


def kohler_zero(wavenumbers, App, w_regions, alpha0=1.25, alpha1=49.95, n_alpha=150, n_components=8):
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
        Q_ext[i][:] = q_ext_kohler(wn, alpha=alpha[i])

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


def q_ext_kohler(wn, alpha):
    """
    Compute the scattering extinction values for a given alpha and a range of wavenumbers
    :param wn: array of wavenumbers
    :param alpha: scalar alpha
    :return: array of scattering extinctions calculated for alpha in the given wavenumbers
    """
    rho = alpha * wn
    Q = 2.0 - (4.0 / rho) * np.sin(rho) + (2.0 / rho) ** 2.0 * (1.0 - np.cos(rho))
    return Q
