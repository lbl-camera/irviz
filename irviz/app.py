from irviz.viewer import Viewer
import h5py as h5
from dask import array as da
import dash
import dash_html_components as html


def open_ir_file(h5_file):
    f = h5.File(h5_file, 'r')
    data = f['irmap']['DATA']['data']
    return da.from_array(data)


if __name__ == "__main__":
    _app = None
    _jupyter_app_kwargs = dict()
    try:
        from jupyter_dash import JupyterDash

        _app = JupyterDash(__name__)
        _jupyter_app_kwargs['mode'] = 'inline'
    except ImportError:
        _app = dash.Dash(__name__, update_title=None)

    data = open_ir_file('C:\\Users\\LBL\\PycharmProjects\\Xi-CAM\\TestData\\STXM\\nexus\\ir_stxm.h5')
    viewer = Viewer(data.compute(), app=_app)
    viewer2 = Viewer(data.compute(), app=_app)

    div = html.Div(children=[viewer, viewer2])
    _app.layout = div

    _app.run_server(debug=True,
                    dev_tools_props_check=False,
                    **_jupyter_app_kwargs)
