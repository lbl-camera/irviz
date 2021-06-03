from irviz.viewer import Viewer
import h5py as h5
from dask import array as da
import dash
import dash_html_components as html
import sklearn.decomposition

TEST_FILE = '/home/ihumphrey/Dev/irviz/data/ir_stxm.h5'

def open_ir_file(h5_file):
    f = h5.File(h5_file, 'r')
    data = f['irmap']['DATA']['data']
    bounds_grid = f['irmap']['DATA']['energy'][:], f['irmap']['DATA']['sample_y'][:], f['irmap']['DATA']['sample_x'][:]
    bounds = list(map(lambda grid: (grid.min(), grid.max()-grid.min()), bounds_grid))
    print(bounds)
    return da.from_array(data), bounds


if __name__ == "__main__":
    _app = None
    _jupyter_app_kwargs = dict()
    try:
        from jupyter_dash import JupyterDash

        _app = JupyterDash(__name__)
        _jupyter_app_kwargs['mode'] = 'inline'
    except ImportError:
        _app = dash.Dash(__name__, update_title=None)

    data, bounds = open_ir_file(TEST_FILE)
    model = sklearn.decomposition.PCA(n_components=3)
    decomposition = model.fit_transform(data.transpose(1,2,0).reshape(-1, data.shape[0])).T.reshape(-1, *data.shape[1:])

    viewer = Viewer(_app, data.compute(), decomposition=decomposition, bounds=bounds)

    div = html.Div(children=[viewer])
    # viewer2 = Viewer(data.compute(), app=_app)
    # div = html.Div(children=[viewer, viewer2])  # TEST for jupyter
    _app.layout = div

    _app.run_server(debug=True,
                    dev_tools_props_check=False,
                    **_jupyter_app_kwargs)
