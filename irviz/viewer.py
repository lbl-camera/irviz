from functools import partial
from typing import List
from bokeh.application import Application
from bokeh.application.handlers import FunctionHandler
from bokeh.models.widgets import Slider, DataTable, Div
from bokeh.models import Span, GridBox
from bokeh.io import show, output_notebook
from bokeh.plotting import figure, curdoc, ColumnDataSource
from bokeh.layouts import column, row, layout, gridplot
from bokeh.events import Tap
import numpy as np
import h5py as h5
from dask import array as da


class Viewer:

    def __init__(self, data, location: List[int] = None):

        self.location = location or list(np.asarray(data.shape)/2)
        self.data = data

        self.main_figure = figure()
        self.main_figure.xaxis.axis_label = 'X'
        self.main_figure.yaxis.axis_label = 'Y'
        self.right_figure = figure()
        self.right_figure.xaxis.axis_label = 'E'
        self.right_figure.yaxis.axis_label = 'Y'
        self.top_figure = figure()
        self.top_figure.xaxis.axis_label = 'X'
        self.top_figure.yaxis.axis_label = 'E'
        self.corner_figure = figure()
        self.corner_figure.xaxis.axis_label = 'E'
        self.corner_figure.yaxis.axis_label = 'I'

        self.main_x_line = Span(location=0, dimension='height', line_color='cyan', line_width=3)
        self.main_y_line = Span(location=0, dimension='width', line_color='cyan', line_width=3)
        self.right_E_line = Span(location=0, dimension='height', line_color='cyan', line_width=3)
        self.right_y_line = Span(location=0, dimension='width', line_color='cyan', line_width=3)
        self.top_E_line = Span(location=0, dimension='width', line_color='cyan', line_width=3)
        self.top_x_line = Span(location=0, dimension='height', line_color='cyan', line_width=3)
        self.corner_E_line = Span(location=0, dimension='height', line_color='cyan', line_width=3)
        self.main_figure.renderers.extend([self.main_y_line, self.main_x_line])
        self.right_figure.renderers.extend([self.right_y_line, self.right_E_line])
        self.top_figure.renderers.extend([self.top_x_line, self.top_E_line])
        self.corner_figure.renderers.append(self.corner_E_line)

        self.spectra_source = ColumnDataSource(data={'E': [], 'I': []})
        self.corner_figure.line(x='E', y='I', source=self.spectra_source)

        self.main_figure.on_event(Tap, partial(self.click_event, axes=[2, 1]))
        self.right_figure.on_event(Tap, partial(self.click_event, axes=[0, 1]))
        self.top_figure.on_event(Tap, partial(self.click_event, axes=[2, 0]))
        self.corner_figure.on_event(Tap, partial(self.click_event, axes=[0]))

        self.corner_figure.on_event(Tap, self.debug)

        self.div = Div()

        self.widget = gridplot([[self.top_figure, self.corner_figure],
                                [self.main_figure, self.right_figure],
                                [self.div]])


        self.show_location(self.location, [0, 1, 2])

    def click_event(self, event, axes):
        x, y = event.x, event.y
        self.show_location([x, y], axes)

    def show_location(self, location, axes):
        for val, axis in zip(location, axes):
            self.location[axis] = val

        bin_location = list(map(int, self.location))

        self.main_figure.image(image=[np.asarray(self.data[bin_location[0]])], x=0, y=0, dw=self.data.shape[2], dh=self.data.shape[1])
        self.right_figure.image(image=[np.asarray(self.data[:, :, bin_location[2]].T)], x=0, y=0, dw=self.data.shape[0], dh=self.data.shape[1])
        self.top_figure.image(image=[np.asarray(self.data[:, bin_location[1]])], x=0, y=0, dw=self.data.shape[2], dh=self.data.shape[0])
        spectra_data = np.asarray(self.data[:, int(bin_location[1]), int(bin_location[2])])
        self.spectra_source.data.update({'E': np.arange(len(spectra_data)),
                                         'I': spectra_data})

        self.main_x_line.location = self.location[2]
        self.main_y_line.location = self.location[1]
        self.right_E_line.location = self.location[0]
        self.right_y_line.location = self.location[1]
        self.top_E_line.location = self.location[0]
        self.top_x_line.location = self.location[2]
        self.corner_E_line.location = self.location[0]

    def debug(self, event):
        self.div.text = f'x: {event.x:.2f}, y: {event.y:.2f}'


def notebook_viewer(data, location=None, theme='dark_minimal'):
    output_notebook()
    #add server-related code inside this modify_doc function
    def modify_doc(doc): #use doc as you use curdoc() in bokeh server
        viewer = Viewer(data, location)
        doc.add_root(viewer.widget)
        doc.theme = theme
    #     doc.on_change(...)
    #     doc.add_periodic_callback(...)

    handler = FunctionHandler(modify_doc)
    app = Application(handler)
    show(app)