import json

import dash
import dash_html_components as html
from dash.dependencies import Input, Output
from dash_slicer import VolumeSlicer
import h5py as h5


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
slicer = VolumeSlicer(app, volume)

slicer.graph.figure.update_xaxes(title_text='blah',
                                 showgrid=True,
                                 showticklabels=True,
                                 tickmode='linear',
                                 tick0=0,
                                 dtick=1)
slicer.graph.figure.update_yaxes(title_text='y axis',
                                 showgrid=True,
                                 showticklabels=True,
                                 tick0=0,
                                 dtick=1)
slicer.graph.figure.update_layout(title="Title",
                                  xaxis_title="X Axis",
                                  yaxis_title="Y Axis",
                                  margin=None)  # {"l": 0, "r": 0, "b": 30, "t": 0, "pad": 10})

app.layout = html.Div(children=[
    slicer.graph,
    html.Div(id='pixel_info'),
    slicer.slider,
    html.Div(id='debug_pixel_info'),
    *slicer.stores
])


@app.callback(Output('pixel_info', 'children'),
              [Input(slicer.graph.id, 'clickData')])
def update_pixel_info(click_data):
    if click_data is not None:
        pixel = click_data['points'][0]
        x = pixel['x']
        y = pixel['y']
        z = pixel['z']['0']  # R of RGBA z value (RGB same since greyscale)
        return f"x: {x}, y: {y}, I: {z}"
    return "x: , y: , I: "

# @app.callback(Output('debug_pixel_info', 'children'),
#               [Input(slicer.graph.id, 'clickData')])
# def debug_update_pixel_info(state):
#     return json.dumps(state, indent=4)


if __name__ == "__main__":
    app.run_server(debug=True, dev_tools_props_check=False)
