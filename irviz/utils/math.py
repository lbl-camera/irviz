import numpy as np


def nearest_bin(x, bounds, bin_count, rounding=True):
    if len(bounds) == 2:
        if rounding:
            return round((x - bounds[0]) / (bounds[1] - bounds[0]) * (bin_count - 1))
        return int((x - bounds[0]) / (bounds[1] - bounds[0]) * (bin_count - 1))
    else:  # for per-bin valued bounds
        return np.abs(np.array(bounds) - x).argmin()
