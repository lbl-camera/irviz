from functools import partial

import dash_core_components as dcc
import numpy as np
from dash.dependencies import Input, Output, ALL
from dask import array as da
from plotly import graph_objects as go

from irviz.utils.dash import targeted_callback
from irviz.utils.math import nearest_bin
__all__ = ['SpectraPlotGraph']


# TODO no annotations outside bounds (check in viewer)
# TODO color picker for annotations in modal dialog (and maybe in the annotations settings layout?)
# TODO (extra): dynamic energy slicer name (name is its value)

class SpectraPlotGraph(dcc.Graph):
    title = 'Spectra Intensities'

    def __init__(self, data, bounds, parent, decomposition=None, component_spectra=None, invert_spectra_axis=False, error_func=None, **kwargs):

        """Interactive Graph that shows spectral intensities at a selectable energy / wave-number index.

        Parameters
        ----------
        data : dask.array
            3D array containing data with axes E (or wave-number), y, and x for displaying in the Graph
        bounds : list or np.ndarray
            Collection that defines the bounds (min & max) pairs for E / Wave-number, y, and x data
            (e.g. a list that contains 3 min/max pairs)
        parent : Component
            Reference to Component that created this Graph (for registering callbacks)
        decomposition : np.ndarray
            (optional) Decomposition of the data
        component_spectra : list or np.ndarray
            List of component spectra from the decomposition
        invert_spectra_axis : bool
            Indicates whether or not to invert the spectra axis (x axis) of the plot (default is False)
        error_func : Callable[[NDArray[(Any, Any)]], np.ndarray[Any]]
            A callable function that takes an array of shape (E, N), where E is the length of the spectral dimension and
            N is the number of curves over which to calculate error. The return value is expected to be a 1-D array of
            length E. The default is to apply a std dev over the N axis.
        kwargs
            Additional keyword arguments to be passed into Graph
        """
        self._data = data
        self._decomposition = decomposition
        self._invert_spectra_axis = invert_spectra_axis
        self._error_func = error_func or partial(np.std, axis=1)
        self._parent = parent
        self._bounds = bounds
        self._component_spectra = np.asarray(component_spectra)

        self.xaxis_title = kwargs.pop('xaxis_title', '')
        self.yaxis_title = kwargs.pop('yaxis_title', '')

        #  default to middle x,y
        _y_index = (self._data.shape[1] - 1) // 2
        _x_index = (self._data.shape[2] - 1) // 2

        y = np.asarray(self._data[:, _y_index, _x_index])
        x = np.linspace(bounds[0][0], bounds[0][1], self._data.shape[0])

        init_x_name = (self._bounds[2][0] + self._bounds[2][1]) / 2
        init_y_name = (self._bounds[1][0] + self._bounds[1][1]) / 2
        self._plot = go.Scattergl(x=x,
                                  y=y,
                                  name=f'spectrum @ {init_x_name:.2f}, {init_y_name:.2f}',
                                  mode='lines')
        self._weighted_sum = go.Scattergl(name=f'weighted component sum',
                                          mode='lines')
        if self._decomposition is not None and self._component_spectra is not None:
            self._weighted_sum.x = x
            self._weighted_sum.y = np.dot(self._decomposition[:, _y_index, _x_index], self._component_spectra)
        self._avg_plot = go.Scattergl(name='selection average',
                                      mode='lines',
                                      legendgroup='_average')
        self._upper_error_plot = go.Scatter(line=dict(width=0),
                                            marker=dict(color="#444"),
                                            hoverinfo='skip',
                                            showlegend=False,
                                            mode='lines',
                                            legendgroup='_average')
        self._lower_error_plot = go.Scatter(line=dict(width=0),
                                            marker=dict(color="#444"),
                                            fill='tonexty',
                                            fillcolor='rgba(68, 68, 68, 0.3)',
                                            showlegend=False,
                                            hoverinfo='skip',
                                            mode='lines',
                                            legendgroup='_average')

        if self._component_spectra.ndim != 2:
            self._component_plots = []
        else:
            self._component_plots = [go.Scattergl(x=self._plot.x,
                                                  y=self._component_spectra[i],
                                                  name=f'Component #{i+1}',
                                                  visible='legendonly',
                                                  legendgroup='_components')
                                     for i in range(self._component_spectra.shape[0])]

        # Define starting point for energy index (for the slicer line trace)
        default_slice_index = (bounds[0][1] + bounds[0][0]) / 2  # estimate
        # Find the closest wavenumber / energy value to use
        default_slice_index = x[np.abs(np.array(x) - default_slice_index).argmin()]

        self._slicer_index = default_slice_index
        self._slicer_name = 'slicer'

        # x coords positioned relative to the x-axis values
        # y coords positioned according to the plot height (0 = bottom, 1.0 = top)
        # self._energy_line = go.layout.Shape(type='line',
        #                                     name='slicer',
        #                                     # width=3,
        #                                     xref='x',
        #                                     yref='paper',
        #                                     x0=default_slice_index,
        #                                     x1=default_slice_index,
        #                                     y0=0,
        #                                     y1=1)

        fig = self._update_figure()

        super(SpectraPlotGraph, self).__init__(id=self._id(),
                                               figure=fig,
                                               className='col-lg-9 p-0',
                                               responsive=True,
                                               style=dict(display='flex',
                                                          flexDirection='row',
                                                          height='100%',
                                                          minHeight='450px'),
                                               )

    def register_callbacks(self):
        # When points are selected on the MapGraph, add additional statistics and components plots
        targeted_callback(self._update_average_plot,
                          Input(self._parent.map_graph.id, 'selectedData'),
                          Output(self.id, 'figure'),
                          app=self._parent._app)

        # On selection of pair-plot points, show their average
        targeted_callback(self._update_average_plot,
                          Input(self._parent.pair_plot_graph.id, 'selectedData'),
                          Output(self.id, 'figure'),
                          app=self._parent._app)

        # When this SpectraGraph itself is clicked, update the energy slicer line
        targeted_callback(self._update_energy_line,
                          Input(self.id, 'clickData'),
                          Output(self.id, 'figure'),
                          app=self._parent._app)

        # When any slice graph is clicked, update plot with the clicked x,y coord
        targeted_callback(self.show_click,
                          Input({'type': 'slice_graph',
                                 'subtype': ALL,
                                 'index': self._parent._instance_index},
                                'clickData'),
                          Output(self.id, 'figure'),
                          app=self._parent._app)

        # Wire-up visibility toggle
        targeted_callback(self._set_visibility,
                          Input(self._parent._graph_toggles.id, 'value'),
                          Output(self.id, 'style'),
                          app=self._parent._app)

        # Chain annotations update to refresh figure
        targeted_callback(self._update_figure,
                          Input(self._parent.spectra_graph_annotations.id, 'children'),
                          Output(self.id, 'figure'),
                          app=self._parent._app)

        # update with slice annotations
        targeted_callback(self.update_annotations,
                          Input(self._parent.slice_graph_annotations.id, 'children'),
                          Output(self.id, 'figure'),
                          app=self._parent._app)

    def show_click(self, click_data):
        y = click_data["points"][0]["y"]
        x = click_data["points"][0]["x"]
        _x_index = nearest_bin(x, self._bounds[2], self._data.shape[2])
        _y_index = nearest_bin(y, self._bounds[1], self._data.shape[1])
        self._plot.y = np.asarray(self._data[:, _y_index, _x_index])

        # update the legend for the spectrum plot
        self._plot.name = f'spectrum @ {x:.2f}, {y:.2f}'

        # update the weighted sum
        if self._decomposition is not None and self._component_spectra is not None:
            self._weighted_sum.x = self._plot.x
            self._weighted_sum.y = np.dot(self._decomposition[:, _y_index, _x_index], self._component_spectra)

        return self._update_figure()

    def update_annotations(self, *_):
        return self._update_figure()

    @property
    def annotations(self):
        annotations = []
        for shape in self.figure.layout.shapes:
            # Ignore the energy slicer line
            if shape.name != self._energy_line.name:
                if shape.visible is not False:  # visible by default set to None instead of True ...
                    annotation = dict(name=shape.name)
                    x0 = shape.x0
                    x1 = shape.x1
                    if x0 == x1:
                        annotation['position'] = x0
                    else:
                        annotation['range'] = (x0, x1)
                    annotations.append(annotation)
                    # x0 = span[0], x1 = span[1], name = name,
                    # fillcolor = color, opacity = 0.2, line_width = 0
                # TODO: support color, etc.
        return annotations

    @property
    def spectrum(self):
        """The currently shown spectrum energy/wavenumber and intensity values"""
        return self._plot.x, self._plot.y

    @property
    def spectral_value(self):
        """The current value of the crosshair position in energy/wavenumber"""
        return self._slicer_index

    @property
    def spectral_index(self):
        """The current index of the crosshair position along the energy/wavenumber domain"""
        return self._plot.x.tolist().index(self._slicer_index)

    @property
    def intensity(self):
        """The intensity value of the crosshair position"""
        intensity_index = self._plot.x.tolist().index(self._slicer_index)
        return self._plot.y[intensity_index]

    @property
    def position(self):
        """The spatial position of the current spectrum"""
        return self._parent.map_graph.position

    @staticmethod
    def _set_visibility(switches_value):
        if 'show_spectra' in switches_value:
            return {'display': 'block'}
        else:
            return {'display': 'none'}

    def add_annotation(self, annotation):
        line_kwargs = {'annotation_position': 'top',
                       'line_dash': 'dot',
                       'opacity': 0.6}
        span = annotation.get('range', None)
        position = annotation.get('position', None)
        color = annotation.get('color', 'gray')
        name = annotation.get('name', 'Unnamed')
        line_kwargs['line'] = {'color': color}

        # # TODO: consider turning off range and position in same annotation
        # # Don't add two annotation texts if we are creating both a vrect and vline
        # if span is not None and position is not None:
        #     fig.add_vrect(x0=span[0], x1=span[1], name=name,
        #                   fillcolor=color, opacity=0.2, line_width=0)
        #     fig.add_vline(x=position, annotation_text=name, **line_kwargs)

        if span is not None:
            self.figure.add_vrect(x0=span[0], x1=span[1], name=name,
                          fillcolor=color, opacity=0.2, line_width=0)
            # Create invisible vline so we can get the text annotation above the middle of the rect range
            center = (span[0] + span[1]) / 2
            self.figure.add_vline(x=center, annotation_text=name, visible=False, **line_kwargs)

        elif position is not None:
            self.figure.add_vline(x=position, name=name, annotation_text=name, **line_kwargs)

    def _traces_from_annotations(self, annotations):
        traces = []

        for annotation in annotations:
            mask = annotation['mask']

            # Dask arrays do fancy indexing differently, and the results have different orientations
            if isinstance(self._data, da.Array):
                y_mask, x_mask = np.nonzero(mask)
                data_slice = self._data.vindex[:, y_mask, x_mask]
                y = np.mean(data_slice, axis=0)
                error = self._error_func(np.asarray(data_slice.T))
            else:
                data_slice = self._data[:, mask]
                y = np.mean(data_slice, axis=1)
                error = self._error_func(np.asarray(data_slice))

            average_trace = go.Scatter(y=y,
                                       x=self._plot.x,
                                       name=f'average @ {annotation["name"]}',
                                       mode='lines',
                                       legendgroup=annotation['name'])
            upper_error_trace = go.Scatter(y=error + self._avg_plot.y,
                                           x=self._plot.x,
                                           line=dict(width=0),
                                           marker=dict(color="#444"),
                                           hoverinfo='skip',
                                           showlegend=False,
                                           mode='lines',
                                           legendgroup=annotation['name'])
            lower_error_trace = go.Scatter(line=dict(width=0),
                                           y=self._avg_plot.y - error,
                                           x=self._plot.x,
                                           marker=dict(color="#444"),
                                           fill='tonexty',
                                           fillcolor='rgba(68, 68, 68, 0.3)',
                                           showlegend=False,
                                           hoverinfo='skip',
                                           mode='lines',
                                           legendgroup=annotation['name'])
            traces.extend([average_trace, upper_error_trace, lower_error_trace])

        return traces

    def _update_figure(self, *_):
        # Get shapes
        current_figure = getattr(self, 'figure', None)
        shapes = []
        annotations = []
        if current_figure is not None:
            shapes = list(filter(lambda shape: shape.visible is not False, current_figure.layout.shapes))  # visible defaults to None?
            shapes.extend(list(filter(lambda shape: shape.visible is False, current_figure.layout.shapes)))
            annotations = current_figure.layout.annotations
        new_figure = go.Figure([self._plot,
                         self._avg_plot,
                         self._weighted_sum,
                         self._upper_error_plot,
                         self._lower_error_plot,
                         *self._traces_from_annotations(self._parent.slice_annotations),
                         *self._component_plots])
        new_figure.update_layout(title=self.title,
                          xaxis_title=self.xaxis_title,
                          yaxis_title=self.yaxis_title)
        if self._invert_spectra_axis:
            new_figure.update_xaxes(autorange="reversed")

        # Always add the slicer line
        new_figure.add_vline(x=self._slicer_index,
                             name=f'{self._slicer_name}',
                             annotation_name=f'{self._slicer_name}',
                             annotation_text=f'{self._slicer_index:.2f}',
                             annotation_position='top')

        for shape in shapes:
            # Ignore slicer line as a user-annotation shape
            if shape.name != self._slicer_name:
                new_figure.add_shape(shape)

        # Add in the plotly annotations (the text above the shapes)
        for annotation in annotations:
            if annotation.name != self._slicer_name:
                new_figure.add_annotation(annotation)

        return new_figure

    def _update_average_plot(self, selected_data):
        if selected_data is not None and len(selected_data['points']) > 0:
            raveled_indexes = list(map(lambda point: point['pointIndex'],
                                       filter(lambda point: point['curveNumber']==0,
                                              selected_data['points'])))
            y_indexes, x_indexes = np.unravel_index(raveled_indexes, self._data.shape[1:])

            # Dask arrays do fancy indexing differently, and the results have different orientations
            if isinstance(self._data, da.Array):
                self._avg_plot.y = np.mean(self._data.vindex[:, y_indexes, x_indexes], axis=0)
                error = self._error_func(np.asarray(self._data.vindex[:, y_indexes, x_indexes].T))
            else:
                self._avg_plot.y = np.mean(self._data[:, y_indexes, x_indexes], axis=1)
                error = self._error_func(np.asarray(self._data[:, y_indexes, x_indexes]))
            self._avg_plot.x = self._plot.x
            self._avg_plot.visible = True

            self._upper_error_plot.x = self._avg_plot.x
            self._upper_error_plot.y = error + self._avg_plot.y
            self._lower_error_plot.x = self._avg_plot.x
            self._lower_error_plot.y = self._avg_plot.y - error
        else:
            self._avg_plot.x = []
            self._avg_plot.y = []
            self._avg_plot.visible = False
            self._upper_error_plot.x = []
            self._upper_error_plot.y = []
            self._lower_error_plot.x = []
            self._lower_error_plot.y = []

        return self._update_figure()

    def _update_energy_line(self, click_data):
        self._slicer_index = click_data["points"][0]["x"]
        return self._update_figure()

    def _id(self):
        return {'type': 'spectraplot',
                'index': self._parent._instance_index}