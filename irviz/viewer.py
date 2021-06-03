from functools import partial
from typing import List
from bokeh.application import Application
from bokeh.application.handlers import FunctionHandler
from bokeh.models.widgets import Slider, DataTable, Div, Tabs, Panel, Dropdown, RadioButtonGroup, Toggle
from bokeh.models import Span, GridBox, ProxyToolbar, ToolbarBox, glyphs
from bokeh.io import show, output_notebook
from bokeh.plotting import figure, curdoc, ColumnDataSource
from bokeh.layouts import column, row, layout, gridplot, grid
from bokeh.events import Tap
import numpy as np

# TODO:
# Bind fig.on_event(SelectionGeometry, ...) to capture lasso selection on images then select appro. points in pair plot


class Viewer:

    def __init__(self, data, decomposition=None, bounds=None, location: List[int] = None):
        # State stash
        self.location = location or list(np.asarray(data.shape)/2)
        self.data = data
        self.decomposition = decomposition
        self.bounds = bounds
        if self.bounds is None:
            self.bounds = list(zip([0, 0, 0], self.data.shape))

        # Build figures
        self.main_figure = figure(title='IR Spectral Map')
        self.main_figure.xaxis.axis_label = 'X'
        self.main_figure.yaxis.axis_label = 'Y'
        self.right_figure = figure(title='Orthogonal Slice (X)', visible=False)
        self.right_figure.xaxis.axis_label = 'E'
        self.right_figure.yaxis.axis_label = 'Y'
        self.top_figure = figure(title='Orthogonal Slice (Y)', visible=False)
        self.top_figure.xaxis.axis_label = 'X'
        self.top_figure.yaxis.axis_label = 'E'
        self.corner_figure = figure(title='IR Spectra')
        self.corner_figure.xaxis.axis_label = 'E'
        self.corner_figure.yaxis.axis_label = 'I'
        self.pair_plot_figure = figure(title='Pair Plot', tools="lasso_select", visible=False)
        self.decomposition_figure = figure(title='Decomposition Maps', visible=False)
        self.decomposition_figure.xaxis.axis_label = 'X'
        self.decomposition_figure.yaxis.axis_label = 'Y'

        # Initialize data sources
        self.spectra_source = ColumnDataSource(data={'E': [], 'I': []})
        self.pair_plot_source = ColumnDataSource(data={'x': [], 'y': []})
        self.main_image_source = ColumnDataSource(data={'image': []})
        self.main_overlay_source = ColumnDataSource(data={'image': []})
        self.right_image_source = ColumnDataSource(data={'image': []})
        self.top_image_source = ColumnDataSource(data={'image': []})
        self.decomposition_image_source = ColumnDataSource(data={'image': []})

        # Initialize glyphs
        self.main_figure.image(image='image', source=self.main_image_source, x=self.bounds[2][0], y=self.bounds[1][0], dw=self.bounds[2][1], dh=self.bounds[1][1])
        self.main_figure.image(image='image', source=self.main_overlay_source, x=self.bounds[2][0], y=self.bounds[1][0], dw=self.bounds[2][1], dh=self.bounds[2][1], global_alpha=.3, palette="OrRd3")
        self.right_figure.image(image='image', source=self.right_image_source, x=self.bounds[0][0], y=self.bounds[1][0], dw=self.bounds[0][1], dh=self.bounds[1][1])
        self.top_figure.image(image='image', source=self.top_image_source, x=self.bounds[2][0], y=self.bounds[0][0], dw=self.bounds[2][1], dh=self.bounds[0][1])
        self.decomposition_figure.image(image='image', source=self.decomposition_image_source, x=self.bounds[2][0], y=self.bounds[1][0], dw=self.bounds[2][1], dh=self.bounds[1][1])
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
        self.corner_figure.line(x='E', y='I', source=self.spectra_source)
        self.pair_plot_figure.circle(x='x', y='y', source=self.pair_plot_source)#, fill_alpha=0.2, size=10)

        # Build proxy toolbar
        figures = [self.main_figure,
                   self.corner_figure,
                   self.right_figure,
                   self.top_figure,
                   self.pair_plot_figure,
                   self.decomposition_figure]
        for f in figures:
            f.toolbar_location = None
        proxy_toolbar = ProxyToolbar(toolbars=list(map(lambda f: f.toolbar, figures)),
                                     tools=sum([f.toolbar.tools for f in figures], []))
        toolbar = ToolbarBox(toolbar=proxy_toolbar, toolbar_location='above')

        # Build widgets
        component_labels = [f'Component #{i+1}' for i in range(decomposition.shape[0])]
        self.decomposition_component = RadioButtonGroup(labels=component_labels, active=0)
        self.pair_plot_component_1 = RadioButtonGroup(labels=component_labels, active=0)
        self.pair_plot_component_2 = RadioButtonGroup(labels=component_labels, active=1)
        self.toggle_spectra_button = Toggle(label='Show Spectra', active=True)
        self.toggle_pair_plot_button = Toggle(label='Show Pair Plot', active=False) #active=decompostion is not None)
        self.toggle_decomposition_button = Toggle(label='Show Decomposition', active=False) #, active=decompostion is not None)
        self.toggle_orthogonal_button = Toggle(label='Show Orthogonal Slices', active=False)

        # Bind events
        self.main_figure.on_event(Tap, partial(self.click_event, axes=[2, 1]))
        self.right_figure.on_event(Tap, partial(self.click_event, axes=[0, 1]))
        self.top_figure.on_event(Tap, partial(self.click_event, axes=[2, 0]))
        self.corner_figure.on_event(Tap, partial(self.click_event, axes=[0]))
        self.decomposition_figure.on_event(Tap, partial(self.click_event, axes=[2, 1]))

        self.decomposition_component.on_change('active', lambda attr, old, new: self.show_component())
        self.toggle_decomposition_button.on_change('active', lambda attr, old, new: setattr(self.decomposition_figure, 'visible', new))
        self.toggle_spectra_button.on_change('active', lambda attr, old, new: setattr(self.corner_figure, 'visible', new))
        self.toggle_pair_plot_button.on_change('active', lambda attr, old, new: setattr(self.pair_plot_figure, 'visible', new))
        self.toggle_orthogonal_button.on_change('active', lambda attr, old, new: (setattr(self.right_figure, 'visible', new),
                                                                                  setattr(self.top_figure, 'visible', new)))

        self.pair_plot_component_1.on_change('active', lambda attr, old, new: self.show_pair_plot())
        self.pair_plot_component_2.on_change('active', lambda attr, old, new: self.show_pair_plot())
        self.pair_plot_source.selected.on_change('indices', self.show_pair_plot_selection)

        # Build layout
        self.location_div = Div()
        self.info_column = column(self.location_div)
        self.info_panel = Panel(child=self.info_column, title="Info")
        self.settings_column = column(self.toggle_spectra_button,
                                      self.toggle_decomposition_button,
                                      self.toggle_pair_plot_button,
                                      self.toggle_orthogonal_button,
                                      Div(text='Decomposition Component'),
                                      self.decomposition_component,
                                      Div(text='Pair Plot Components'),
                                      self.pair_plot_component_1,
                                      self.pair_plot_component_2)
        self.settings_panel = Panel(child=self.settings_column, title="Settings")
        self.control_tabs = Tabs(tabs=[self.settings_panel, self.info_panel])

        # Size figures
        self.corner_figure.plot_width = self.main_figure.plot_width * 2

        # Todo: implement merged toolbars
        self.widget = layout([[toolbar],
                              [self.main_figure, self.control_tabs],
                              [self.top_figure, self.right_figure],
                              [self.corner_figure],
                              [self.decomposition_figure, self.pair_plot_figure]])

        # Initialize views
        self.show_location(self.location, [0, 1, 2])
        self.show_component()
        self.show_pair_plot()

    @property
    def bin_location(self):
        return tuple(map(int, self.location))

    def click_event(self, event, axes):
        x, y = event.x, event.y
        self.show_location([x, y], axes)

    def show_location(self, location, axes):
        for val, axis in zip(location, axes):
            self.location[axis] = val
            self.print_location()

        self.main_image_source.data.update({'image': [np.asarray(self.data[self.bin_location[0]])]})
        self.right_image_source.data.update({'image': [np.asarray(self.data[:, :, self.bin_location[2]].T)]})
        self.top_image_source.data.update({'image': [np.asarray(self.data[:, self.bin_location[1]])]})
        spectra_data = np.asarray(self.data[:, int(self.bin_location[1]), int(self.bin_location[2])])
        self.spectra_source.data.update({'E': np.linspace(self.bounds[0][0], self.bounds[0][0]+self.bounds[0][1], len(spectra_data)),
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
        self.decomposition_image_source.data.update({'image':[np.asarray(self.decomposition[self.decomposition_component.active])]})

    def show_pair_plot(self):
        self.pair_plot_source.data.update({'x': np.asarray(self.decomposition[self.pair_plot_component_1.active].ravel()),
                                           'y': np.asarray(self.decomposition[self.pair_plot_component_2.active].ravel())})
        self.pair_plot_figure.xaxis.axis_label = self.pair_plot_component_1.labels[self.pair_plot_component_1.active]
        self.pair_plot_figure.yaxis.axis_label = self.pair_plot_component_2.labels[self.pair_plot_component_2.active]

    def show_pair_plot_selection(self, attr, old, new):
        if old != new:
            mask = np.ones(self.data[0].shape)
            mask.ravel()[new] = 0
            self.main_overlay_source.data.update({'image': [mask]})

    def print_location(self):
        self.location_div.text = f'x: {self.location[2]:.2f}, ' \
                                 f'y: {self.location[1]:.2f}, ' \
                                 f'E: {self.location[0]:.2f}, ' \
                                 f'I: {self.data[self.bin_location[0], self.bin_location[1], self.bin_location[2]].compute():.2f}'


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
