import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import plotly.graph_objects as go
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
from dash_slicer import VolumeSlicer

from irviz.utils import open_ir_file


app = dash.Dash(__name__, update_title=None)

# Create different 2D volume slicers, one for each axis (z, y, x)
volume = open_ir_file('/home/ihumphrey/Dev/xicam/data/ir_stxm.h5')
z_slicer = VolumeSlicer(app, volume, axis=0)
y_slicer = VolumeSlicer(app, volume, axis=1)
x_slicer = VolumeSlicer(app, volume, axis=2)

# Override some of the layout changes that VolumeSlicer does
common_opts = {'xaxis_showticklabels': True,
               'yaxis_showticklabels': True,
               'xaxis_showgrid': True,
               'yaxis_showgrid': True,
               'margin': None}
z_slicer.graph.figure.update_layout(title="y(x)",
                                    xaxis_title="X",
                                    yaxis_title="Y",
                                    # xaxis_tick0=0,
                                    # xaxis_dtick=1,
                                    # yaxis_tick0=0,
                                    # xaxis_dtick=1,
                                    **common_opts)  # {"l": 0, "r": 0, "b": 30, "t": 0, "pad": 10})
y_slicer.graph.figure.update_layout(title="E(y)",
                                    # xaxis_range=[0, volume.shape[2]],
                                    xaxis_scaleanchor="y",
                                    xaxis_scaleratio=(volume.shape[0] / volume.shape[2]),
                                    xaxis_title="X",
                                    yaxis_title="Z",
                                    **common_opts
                                    )
x_slicer.graph.figure.update_layout(title="E(x)",
                                    xaxis_scaleanchor="y",
                                    xaxis_scaleratio=(volume.shape[0] / volume.shape[1]),
                                    xaxis_title="Y",
                                    yaxis_title="Z",
                                    **common_opts)

# Describe our html elements for each slicer
z_view = html.Div([
    z_slicer.graph,
    z_slicer.slider,
    html.Div(id='pixel_info'),
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
            'gridTemplateColumns': '33% 33% 33%',
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
              Input(z_slicer.graph.id, 'clickData'))
def update_pixel_info(click_data):
    if click_data is not None:
        pixel = click_data['points'][0]
        x = pixel['x']
        y = pixel['y']
        z = pixel['z']['0']  # R of RGBA z value (RGB same since greyscale)
        return f"DEBUG: x: {x}, y: {y}, I: {z}"
    return "DEBUG: x: , y: , I: "


@app.callback(Output('spectra_plot', 'figure'),
              Input(z_slicer.state.id, 'data'),
              Input(y_slicer.state.id, 'data'),
              Input(x_slicer.state.id, 'data'))
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
