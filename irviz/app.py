import dash
import dash_html_components as html
from dash_slicer import VolumeSlicer
import h5py as h5


def open_ir_file(h5_file):
    f = h5.File(h5_file, 'r')
    return f['irmap']['DATA']['data'][:]


app = dash.Dash(__name__, update_title=None)

volume = open_ir_file('/home/ihumphrey/Dev/xicam/data/ir_stxm.h5')
slicer = VolumeSlicer(app, volume)

app.layout = html.Div(children=[slicer.graph, slicer.slider, *slicer.stores])


if __name__ == "__main__":
    app.run_server(debug=True, dev_tools_props_check=False)