import h5py as h5


def open_ir_file(h5_file):
    f = h5.File(h5_file, 'r')
    return f['irmap']['DATA']['data'][:]
