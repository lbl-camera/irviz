from bokeh.models.widgets import Slider, DataTable, Div
from bokeh.models import Span
from bokeh.plotting import figure, show, output_notebook, curdoc, ColumnDataSource
from bokeh.layouts import column, row
from bokeh.events import Tap
import numpy as np
import h5py as h5
from dask import array as da


# TODO:
# Coordinates + intensity on click
# 1D ortho
#

pos = [0, 0, 0]

def open_ir_file(h5_file):
    f = h5.File(h5_file, 'r')
    data = f['irmap']['DATA']['data']
    return da.from_array(data)

data = open_ir_file('C:\\Users\\LBL\\PycharmProjects\\Xi-CAM\\TestData\\STXM\\nexus\\ir_stxm.h5')

fig = figure()
ortho_line = Span(location=0, dimension='height', line_color='red', line_width=3)
fig.renderers.append(ortho_line)

ortho_fig = figure()

spectra_source = ColumnDataSource(data={'x':np.arange(data.shape[0]), 'y':np.asarray(data[:, 0, 0])})
spectra_plot = figure()
spectra_line = spectra_plot.line(x='x', y='y', source=spectra_source)

def show_frame(attr, old, i):
    global pos
    pos[0] = i
    fig.image(image=[np.asarray(data[pos[0]])], x=0, y=0, dw=data.shape[1], dh=data.shape[2])

def show_frame_ortho(attr, old, j):
    global pos
    pos[1] = j
    ortho_fig.image(image=[np.asarray(data[:, pos[1]].T)], x=0, y=0, dw=data.shape[0], dh=data.shape[2])
    ortho_line.location = j

show_frame(None, None, 0)
show_frame_ortho(None, None, 0)

slider = Slider(start=0,
                end=data.shape[0]-1,   # the number of slices
                value=0, step=1,
                title='Axis 1')
ortho_slider = Slider(start=0,
                end=data.shape[1]-1,   # the number of slices
                value=0, step=1,
                title='Axis 2')
# slider.on_change('value_throttled', show_frame)
# ortho_slider.on_change('value_throttled', show_frame_ortho)
slider.on_change('value', show_frame)
ortho_slider.on_change('value', show_frame_ortho)

div = Div()

def figure_clicked(event):
    x, y = event.x, event.y
    div.text = f'X: {x}, Y: {y}, I: {data[int(x), int(y), 0].compute()}'
    spectra_data = np.asarray(data[:, int(x), int(y)])
    spectra_source.data['x'] = np.arange(len(spectra_data))
    spectra_source.data['y'] = spectra_data

fig.on_event(Tap, figure_clicked)

# A datasource bound to this datatable can be bound to emit events when runs are selected

curdoc().add_root(row(column(fig, slider, div), column(ortho_fig, ortho_slider), spectra_plot))
