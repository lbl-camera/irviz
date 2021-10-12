import numpy as np


def nearest_bin(x, bounds, bin_count, rounding=True):
    if len(bounds) == 2:
        if rounding:
            return round((x - bounds[0]) / (bounds[1] - bounds[0]) * (bin_count - 1))
        return int((x - bounds[0]) / (bounds[1] - bounds[0]) * (bin_count - 1))
    else:  # for per-bin valued bounds
        return np.abs(np.array(bounds) - x).argmin()


def array_from_selection(selection, shape):
    # Check two cases:
    #     1. selection is None: initial state (no selection) or user has dbl-clicked w/ lasso/selection tool
    #     2. selection['points'] is empty: user has selected no points

    if selection is not None and len(selection['points']) > 0:
        # Get x,y from the raveled indexes
        raveled_indexes = list(map(lambda point: point['pointIndex'],
                                   filter(lambda point: point['curveNumber'] == 0,
                                          selection['points'])))
        mask = np.zeros(shape)
        # Cannot be 0s - must be NaNs (eval to None) so it doesn't affect underlying HeatMap
        mask.fill(np.NaN)
        mask.ravel()[raveled_indexes] = 1
        # Create overlay
        return mask
    else:
        return np.ones(shape) * np.NaN
