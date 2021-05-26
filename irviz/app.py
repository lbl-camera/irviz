import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
from dash_slicer import VolumeSlicer
import h5py as h5
import plotly.graph_objects as go
import numpy as np

#  ***
#  Add vertical spanner (to show how orthogonal view is being sliced)
#  Understand the slicer module


def open_ir_file(h5_file):
    f = h5.File(h5_file, 'r')
    return f['irmap']['DATA']['data'][:]


app = dash.Dash(__name__, update_title=None)

# Create different 2D volume slicers, one for each axis (z, y, x)
volume = open_ir_file('/home/ihumphrey/Dev/xicam/data/ir_stxm.h5')
z_slicer = VolumeSlicer(app, volume, axis=0)
y_slicer = VolumeSlicer(app, volume, axis=1)
x_slicer = VolumeSlicer(app, volume, axis=2)

# Override some of the layout changes that VolumeSlicer does
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

# Describe our html elements for each slicer "view"
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
# Describe html layout for spectral plot
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
    # We need all the slicer state data ready; otherwise, don't update any Graph objects
    if z_data is None or y_data is None or x_data is None:
        raise PreventUpdate

    y_index = y_data['index']
    x_index = x_data['index']
    y = volume[:, y_index, x_index]
    x = np.arange(0, volume.shape[0])
    fig = go.Figure(data={'type': 'scatter', 'x': x, 'y': y})
    fig.update_layout(title=f'Spectra Intensities @ (x: {x_index}, y: {y_index})',
                      xaxis_title="Spectra",
                      yaxis_title="Intensity")
    return fig


if __name__ == "__main__":
    app.run_server(debug=True, dev_tools_props_check=False)
