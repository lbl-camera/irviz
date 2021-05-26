import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from dash_slicer import VolumeSlicer
import h5py as h5
import plotly.graph_objects as go
import numpy as np


# TODO:
#  orthogonal views
#    - 1 2D orthogonal view
#    - 1 1D orthogonal view
#  callback implemented on pixel selection (clicked)
#    - update orthogonal views
#    - display I, coords, etc.
#  ***
#  Add vertical spanner (to show how orthogonal view is being sliced)
#  Understand the slicer module

def open_ir_file(h5_file):
    f = h5.File(h5_file, 'r')
    return f['irmap']['DATA']['data'][:]


app = dash.Dash(__name__, update_title=None)

volume = open_ir_file('/home/ihumphrey/Dev/xicam/data/ir_stxm.h5')
z_slicer = VolumeSlicer(app, volume, axis=0)
y_slicer = VolumeSlicer(app, volume, axis=1)
x_slicer = VolumeSlicer(app, volume, axis=2)

axes_opts = {'showgrid': True, 'showticklabels': True, 'tick0': 0, 'dtick': 1}

z_slicer.graph.figure.update_layout(title="Title",
                                    xaxis_title="X Axis",
                                    yaxis_title="Y Axis",
                                    xaxis=axes_opts,
                                    yaxis=axes_opts,
                                    margin=None)  # {"l": 0, "r": 0, "b": 30, "t": 0, "pad": 10})
y_slicer.graph.figure.update_layout(title="Y Slice",
                                    xaxis_title="X",
                                    yaxis_title="Z",
                                    xaxis_showticklabels=True,
                                    yaxis_showticklabels=True,
                                    xaxis_showgrid=True,
                                    yaxis_showgrid=True,
                                    margin=None)
x_slicer.graph.figure.update_layout(title="X Slice",
                                    xaxis_title="Y",
                                    yaxis_title="Z",
                                    xaxis_showticklabels=True,
                                    yaxis_showticklabels=True,
                                    xaxis_showgrid=True,
                                    yaxis_showgrid=True,
                                    margin=None)

z_view = html.Div([
    z_slicer.graph,
    html.Div(id='pixel_info'),
    z_slicer.slider,
    html.Div(id='debug_pixel_info'),
    *z_slicer.stores
])
y_view = html.Div([
    y_slicer.graph,
    y_slicer.slider,
    *y_slicer.stores
])
x_view = html.Div([
    x_slicer.graph,
    x_slicer.slider,
    *x_slicer.stores
])

spectra_plot = html.Div([
    dcc.Graph(id='spectra_plot')
])

app.layout = html.Div([
    html.Div(
        style={
            'display': 'grid',
            'gridTemplateColumns': '40% 30% 30%',
            'gridTemplateRows': 'auto auto'
        },
        children=[
            z_view,
            y_view,
            x_view
        ]),
    html.Div(
        spectra_plot
    )
])


@app.callback(Output('pixel_info', 'children'),
              [Input(z_slicer.graph.id, 'clickData')])
def update_pixel_info(click_data):
    if click_data is not None:
        pixel = click_data['points'][0]
        x = pixel['x']
        y = pixel['y']
        z = pixel['z']['0']  # R of RGBA z value (RGB same since greyscale)
        return f"x: {x}, y: {y}, I: {z}"
    return "x: , y: , I: "


@app.callback(Output('spectra_plot', 'figure'),
              [Input(z_slicer.state.id, 'data'),
               Input(y_slicer.state.id, 'data'),
               Input(x_slicer.state.id, 'data')])
def do_thing(z_data, y_data, x_data):
    global volume
    if z_data and y_data and x_data:
        y_index = y_data['index']
        x_index = x_data['index']
        y = volume[:, y_index, x_index]
        x = np.arange(0, volume.shape[0])
    else:
        y_index = int(volume.shape[1] / 2)
        x_index = int(volume.shape[2] / 2)
        y = volume[:, y_index, x_index]
        x = np.arange(0, volume.shape[0])

    fig = go.Figure(data={'type': 'scatter', 'x': x, 'y': y})
    fig.update_layout(title=f'Spectra Intensities @ (x: {x_index}, y: {y_index})',
                      xaxis_title="Spectra",
                      yaxis_title="Intensity")


if __name__ == "__main__":
    app.run_server(debug=True, dev_tools_props_check=False)
