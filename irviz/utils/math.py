def nearest_bin(x, bounds, bin_count, rounding=True):
    if round:
        return round((x-bounds[0])/(bounds[1]-bounds[0])*bin_count)
    return int((x-bounds[0])/(bounds[1]-bounds[0])*bin_count)