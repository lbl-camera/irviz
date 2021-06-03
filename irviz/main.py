from bokeh.plotting import figure, show, output_notebook, curdoc, ColumnDataSource
from irviz.viewer import Viewer
import h5py as h5
from dask import array as da
import sklearn.decomposition


# TODO:
# https://docs.bokeh.org/en/latest/docs/user_guide/server.html#directory-format
#


def open_ir_file(h5_file):
    f = h5.File(h5_file, 'r')
    data = f['irmap']['DATA']['data']
    bounds_grid = f['irmap']['DATA']['energy'][:], f['irmap']['DATA']['sample_y'][:], f['irmap']['DATA']['sample_x'][:]
    bounds = list(map(lambda grid: (grid.min(), grid.max()-grid.min()), bounds_grid))
    print(bounds)
    return da.from_array(data), bounds

data, bounds = open_ir_file('C:\\Users\\LBL\\PycharmProjects\\Xi-CAM\\TestData\\STXM\\nexus\\ir_stxm.h5')
model = sklearn.decomposition.PCA(n_components=3)
decomposition = model.fit_transform(data.transpose(1,2,0).reshape(-1, data.shape[0])).T.reshape(-1, *data.shape[1:])

viewer = Viewer(data, decomposition=decomposition) # bounds=bounds
curdoc().add_root(viewer.widget)
curdoc().theme = 'dark_minimal'
