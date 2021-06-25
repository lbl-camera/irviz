import dash
import dash_bootstrap_components as dbc
import dash_html_components as html
import h5py as h5
import numpy as np
import sklearn.decomposition
from dask import array as da
from PIL import Image
from dash_bootstrap_templates import load_figure_template

from irviz.viewer import Viewer

TEST_FILE = '/home/ihumphrey/Dev/irviz/data/ir_stxm.h5'
TEST_FILE = '/home/ihumphrey/Dev/irviz/data/BP-area3a.h5'
OPTICAL_TEST_FILE = 'E:\\BP-area3a_clean.JPG'
# TEST_FILE = '/home/ihumphrey/Dev/irviz/data/BP-area3a.h5'


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


if __name__ == "__main__":
    from irviz.utils import dash as irdash

    load_figure_template("darkly")

    app_kwargs = {'external_stylesheets': [dbc.themes.DARKLY]}
    _jupyter_app_kwargs = dict()
    try:
        from jupyter_dash import JupyterDash

        irdash.app = JupyterDash(__name__, **app_kwargs)
        _jupyter_app_kwargs['mode'] = 'inline'
    except ImportError:
        irdash.app = dash.Dash(__name__, update_title=None, **app_kwargs)

    # data, bounds = open_ir_file(TEST_FILE)
    data, bounds = open_map_file(TEST_FILE)
    # optical = np.flipud(np.average(open_optical_file(OPTICAL_TEST_FILE), axis=2))
    optical = np.flipud(open_optical_file(OPTICAL_TEST_FILE))
    model = sklearn.decomposition.PCA(n_components=3)
    decomposition = model.fit_transform(data.transpose(1,2,0).reshape(-1, data.shape[0])).T.reshape(-1, *data.shape[1:])
    cluster_labels = np.argmax(decomposition, axis=0)
    cluster_label_names = ['Alpha', 'Bravo', 'Charlie']

    viewer = Viewer(irdash.app,
                    data,
                    optical=optical,
                    decomposition=decomposition,
                    bounds=bounds,
                    component_spectra=model.components_,
                    spectra_axis_title='Wavenumber (cm⁻¹)',
                    intensity_axis_title='Intensity',
                    x_axis_title='X (μm)',
                    y_axis_title='Y (μm)',
                    invert_spectra_axis=True,
                    cluster_labels=cluster_labels,
                    cluster_label_names=cluster_label_names,
                    annotations={
                        'x': {
                            'range': (1000, 1500),
                            'color': 'green'
                        },
                        'y': {
                            'position': 300,
                            'range': [200, 500]
                        },
                        'z': {
                            'position': 900,
                            'color': '#34afdd'
                        }
                    }
                    )

    # Testing None decomposition
    # viewer = Viewer(_app, data.compute(), decomposition=None, bounds=bounds)

    div = html.Div(children=[viewer])
    # viewer2 = Viewer(data.compute(), app=_app)
    # div = html.Div(children=[viewer, viewer2])  # TEST for jupyter
    irdash.app.layout = div

    irdash.app.run_server(debug=True,
                          # dev_tools_props_check=False,
                          **_jupyter_app_kwargs)
