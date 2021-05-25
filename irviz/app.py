import dash
import dash_html_components as html
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

app.layout = html.Div(children=[slicer.graph, slicer.slider, *slicer.stores])


if __name__ == "__main__":
    app.run_server(debug=True, dev_tools_props_check=False)
