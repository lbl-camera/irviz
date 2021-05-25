import dash
import dash_html_components as html
from dash_slicer import VolumeSlicer
import h5py as h5
import numpy as np


app = dash.Dash(__name__, update_title=None)

# data = h5.File('example.hdf5', 'r')

volume = np.random.rand(3, 100, 100)
slicer = VolumeSlicer(app, volume)

app.layout = html.Div(children=[slicer.graph, slicer.slider, *slicer.stores])


if __name__ == "__main__":
    app.run_server(debug=True, dev_tools_props_check=False)