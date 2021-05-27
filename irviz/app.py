from bokeh.plotting import figure, show, output_notebook, curdoc, ColumnDataSource
from irviz.viewer import Viewer
import h5py as h5
from dask import array as da

# TODO:
# https://docs.bokeh.org/en/latest/docs/user_guide/server.html#directory-format
#


def open_ir_file(h5_file):
    f = h5.File(h5_file, 'r')
    data = f['irmap']['DATA']['data']
    return da.from_array(data)

data = open_ir_file('C:\\Users\\LBL\\PycharmProjects\\Xi-CAM\\TestData\\STXM\\nexus\\ir_stxm.h5')


viewer = Viewer(data)
curdoc().add_root(viewer.widget)
curdoc().theme = 'dark_minimal'
