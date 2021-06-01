from functools import partial
from typing import List
from bokeh.application import Application
from bokeh.application.handlers import FunctionHandler
from bokeh.models.widgets import Slider, DataTable, Div, Tabs, Panel, Dropdown, RadioButtonGroup, Toggle
from bokeh.models import Span, GridBox, ProxyToolbar, ToolbarBox
from bokeh.io import show, output_notebook
from bokeh.plotting import figure, curdoc, ColumnDataSource
from bokeh.layouts import column, row, layout, gridplot, grid
from bokeh.events import Tap
import numpy as np
import h5py as h5
from dask import array as da


class Viewer:

    def __init__(self, data, decompostion=None, location: List[int] = None):

        self.location = location or list(np.asarray(data.shape)/2)
        self.data = data
        self.decomposition = decompostion

        # Build figures
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
        self.pair_plot_figure = figure()
        self.decomposition_figure = figure()
        self.decomposition_figure.xaxis.axis_label = 'X'
        self.decomposition_figure.yaxis.axis_label = 'Y'

        # Initialize glyphs
        self.main_x_line = Span(location=0, dimension='height', line_color='cyan', line_width=3)
        self.main_y_line = Span(location=0, dimension='width', line_color='cyan', line_width=3)
        self.right_E_line = Span(location=0, dimension='height', line_color='cyan', line_width=3)
        self.right_y_line = Span(location=0, dimension='width', line_color='cyan', line_width=3)
        self.top_E_line = Span(location=0, dimension='width', line_color='cyan', line_width=3)
        self.top_x_line = Span(location=0, dimension='height', line_color='cyan', line_width=3)
        self.corner_E_line = Span(location=0, dimension='height', line_color='cyan', line_width=3)
        self.decomposition_x_line = Span(location=0, dimension='height', line_color='cyan', line_width=3)
        self.decomposition_y_line = Span(location=0, dimension='width', line_color='cyan', line_width=3)
        self.main_figure.renderers.extend([self.main_y_line, self.main_x_line])
        self.right_figure.renderers.extend([self.right_y_line, self.right_E_line])
        self.top_figure.renderers.extend([self.top_x_line, self.top_E_line])
        self.corner_figure.renderers.append(self.corner_E_line)
        self.decomposition_figure.renderers.extend([self.decomposition_x_line, self.decomposition_y_line])

        # Initialize data sources
        self.spectra_source = ColumnDataSource(data={'E': [], 'I': []})
        self.corner_figure.line(x='E', y='I', source=self.spectra_source)
        self.pair_plot_source = ColumnDataSource(data={'x': [], 'y': []})
        self.pair_plot_figure.circle(x='x', y='y', source=self.pair_plot_source)#, fill_alpha=0.2, size=10)

        # Build proxy toolbar
        figures = [self.main_figure,
                   self.corner_figure,
                   self.right_figure,
                   self.top_figure,
                   self.pair_plot_figure,
                   self.decomposition_figure]
        for f in figures:
            figure.toolbar_location = None
        proxy_toolbar = ProxyToolbar(toolbars=list(map(lambda f: f.toolbar, figures)),
                                     tools=sum([ f.toolbar.tools for f in figures ], []))
        toolbar = ToolbarBox(toolbar=proxy_toolbar, toolbar_location='above')

        # Build widgets
        component_labels = [f'Component #{i+1}' for i in range(decompostion.shape[0])]
        self.decomposition_component = RadioButtonGroup(labels=component_labels, active=0)
        self.pair_plot_component_1 = RadioButtonGroup(labels=component_labels, active=0)
        self.pair_plot_component_2 = RadioButtonGroup(labels=component_labels, active=1)
        self.toggle_spectra_button = Toggle(label='Show Spectra', active=True)
        self.toggle_pair_plot_button = Toggle(label='Show Pair Plot', active=decompostion is not None)
        self.toggle_decomposition_button = Toggle(label='Show Decomposition', active=decompostion is not None)
        self.toggle_orthogonal_button = Toggle(label='Show Orthogonal Slices', active=True)

        # Bind events
        self.main_figure.on_event(Tap, partial(self.click_event, axes=[2, 1]))
        self.right_figure.on_event(Tap, partial(self.click_event, axes=[0, 1]))
        self.top_figure.on_event(Tap, partial(self.click_event, axes=[2, 0]))
        self.corner_figure.on_event(Tap, partial(self.click_event, axes=[0]))
        self.decomposition_figure.on_event(Tap, partial(self.click_event, axes=[2, 1]))

        self.corner_figure.on_event(Tap, self.debug)

        self.decomposition_component.on_change('active', lambda attr, old, new: self.show_component())
        self.toggle_decomposition_button.on_change('active', lambda attr, old, new: setattr(self.decomposition_figure, 'visible', new))
        self.toggle_spectra_button.on_change('active', lambda attr, old, new: setattr(self.corner_figure, 'visible', new))
        self.toggle_pair_plot_button.on_change('active', lambda attr, old, new: setattr(self.pair_plot_figure, 'visible', new))
        # self.toggle_orthogonal_button.on_change('active', lambda attr, old, new: setattr(self.pair_plot_figure, 'visible', new))

        self.pair_plot_component_1.on_change('active', lambda attr, old, new: self.show_pair_plot())
        self.pair_plot_component_2.on_change('active', lambda attr, old, new: self.show_pair_plot())
        self.pair_plot_figure.on_event()


        # Build layout
        self.div = Div()
        self.info_column = column(self.div)
        self.info_panel = Panel(child=self.info_column, title="Info")
        self.settings_column = column(self.decomposition_component,
                                      self.toggle_spectra_button,
                                      self.toggle_decomposition_button,
                                      self.toggle_pair_plot_button,
                                      self.toggle_orthogonal_button,
                                      self.pair_plot_component_1,
                                      self.pair_plot_component_2)
        self.settings_panel = Panel(child=self.settings_column, title="Settings")
        self.control_tabs = Tabs(tabs=[self.settings_panel, self.info_panel])

        # Size figures
        self.corner_figure.plot_width = self.main_figure.plot_width * 2

        # Todo: implement merged toolbars
        self.widget = layout([[toolbar],
                              [self.main_figure, self.control_tabs],
                              [self.corner_figure],
                              [self.pair_plot_figure, self.decomposition_figure]])

        # Initialize views
        self.show_location(self.location, [0, 1, 2])
        self.show_component()
        self.show_pair_plot()

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
        self.decomposition_x_line.location = self.location[2]
        self.decomposition_y_line.location = self.location[1]

    def show_component(self):
        self.decomposition_figure.image(image=[np.asarray(self.decomposition[self.decomposition_component.active])], x=0, y=0, dw=self.data.shape[2], dh=self.data.shape[1])

    def show_pair_plot(self):
        self.pair_plot_source.data.update({'x': np.asarray(self.decomposition[self.pair_plot_component_1.active].ravel()),
                                           'y': np.asarray(self.decomposition[self.pair_plot_component_2.active].ravel())})
        self.pair_plot_figure.xaxis.axis_label = self.pair_plot_component_1.labels[self.pair_plot_component_1.active]
        self.pair_plot_figure.yaxis.axis_label = self.pair_plot_component_2.labels[self.pair_plot_component_2.active]

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
