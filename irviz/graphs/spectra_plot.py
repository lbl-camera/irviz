from functools import partial

import dash_core_components as dcc
import numpy as np
from dash.dependencies import Input, Output, ALL
from dask import array as da
from plotly import graph_objects as go

from irviz.utils.dash import targeted_callback
from irviz.utils.math import nearest_bin
__all__ = ['SpectraPlotGraph']


class SpectraPlotGraph(dcc.Graph):
    title = 'Spectra Intensities'

    def __init__(self, data, bounds, parent, component_spectra=None, invert_spectra_axis=False, annotations=None, error_func=None, **kwargs):
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
        component_spectra : list or np.ndarray
            List of component spectra from the decomposition
        invert_spectra_axis : bool
            Indicates whether or not to invert the spectra axis (x axis) of the plot (default is False)
        annotations : dict[str, dict]
            Dictionary that contains annotation names that map to annotations.
            The annotation dictionaries support the following keys:
                'range' : list or tuple of length 2
                'position' : number
                'color' : color (hex str, rgb str, hsl str, hsv str, named CSS color)
            Example:
                annotations={
                    'x': {
                        'range': (1000, 1500),
                        'color': 'green'
                    },
                    'y': {
                        'position': 300,
                        'range': [200, 500]
                    },
                    'z': {
                        'position': 900,
                        'color': '#34afdd'
                    }
                }
        error_func : Callable[[NDArray[(Any, Any)]], np.ndarray[Any]]
            A callable function that takes an array of shape (E, N), where E is the length of the spectral dimension and
            N is the number of curves over which to calculate error. The return value is expected to be a 1-D array of
            length E. The default is to apply a std dev over the N axis.
        kwargs
            Additional keyword arguments to be passed into Graph
        """
        self._data = data
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
        self._avg_plot = go.Scattergl(name='average',
                                      mode='lines')
        self._upper_error_plot = go.Scatter(line=dict(width=0),
                                            marker=dict(color="#444"),
                                            hoverinfo='skip',
                                            showlegend=False,
                                            mode='lines')
        self._lower_error_plot = go.Scatter(line=dict(width=0),
                                            marker=dict(color="#444"),
                                            fill='tonexty',
                                            fillcolor='rgba(68, 68, 68, 0.3)',
                                            showlegend=False,
                                            hoverinfo='skip',
                                            mode='lines')

        if self._component_spectra.ndim != 2:
            self._component_plots = []
        else:
            self._component_plots = [go.Scattergl(x=self._plot.x,
                                                  y=self._component_spectra[i],
                                                  name=f'Component #{i+1}',
                                                  visible='legendonly')
                                     for i in range(self._component_spectra.shape[0])]

        # Define starting point for energy index (for the slicer line trace)
        default_slice_index = (bounds[0][1] + bounds[0][0]) / 2  # estimate
        # Find the closest wavenumber / energy value to use
        default_slice_index = x[np.abs(np.array(x) - default_slice_index).argmin()]

        # x coords positioned relative to the x-axis values
        # y coords positioned according to the plot height (0 = bottom, 1.0 = top)
        self._energy_line = go.layout.Shape(type='line',
                                            # width=3,
                                            xref='x',
                                            yref='paper',
                                            x0=default_slice_index,
                                            x1=default_slice_index,
                                            y0=0,
                                            y1=1)

        # Handle annotations
        self._annotations = annotations

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

    def show_click(self, click_data):
        y = click_data["points"][0]["y"]
        x = click_data["points"][0]["x"]
        _x_index = nearest_bin(x, self._bounds[2], self._data.shape[2])
        _y_index = nearest_bin(y, self._bounds[1], self._data.shape[1])
        self._plot.y = np.asarray(self._data[:, _y_index, _x_index])

        # update the legend for the spectrum plot
        self._plot.name = f'spectrum @ {x:.2f}, {y:.2f}'

        return self._update_figure()

    @property
    def spectrum(self):
        """The currently shown spectrum energy/wavenumber and intensity values"""
        return self._plot.x, self._plot.y

    @property
    def spectral_value(self):
        """The current value of the crosshair position in energy/wavenumber"""
        return self._energy_line.x0

    @property
    def spectral_index(self):
        """The current index of the crosshair position along the energy/wavenumber domain"""
        return self._plot.x.tolist().index(self._energy_line.x0)

    @property
    def intensity(self):
        """The intensity value of the crosshair position"""
        intensity_index = self._plot.x.tolist().index(self._energy_line.x0)
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

    def _add_annotations(self, fig):
        if self._annotations is not None:
            line_kwargs = {'annotation_position': 'top',
                           'line_dash': 'dot',
                           'opacity': 0.6}
            for name, annotation in self._annotations.items():
                span = annotation.get('range', None)
                position = annotation.get('position', None)
                color = annotation.get('color', 'gray')
                line_kwargs['line'] = {'color': color}

                # Don't add two annotation texts if we are creating both a vrect and vline
                if span is not None and position is not None:
                    fig.add_vrect(x0=span[0], x1=span[1],
                                  fillcolor=color, opacity=0.2, line_width=0)
                    fig.add_vline(x=position, annotation_text=name, **line_kwargs)

                elif span is not None:
                    fig.add_vrect(x0=span[0], x1=span[1],
                                  fillcolor=color, opacity=0.2, line_width=0)
                    # Create invisible vline so we can get the text annotation above the middle of the rect range
                    center = (span[0] + span[1]) / 2
                    fig.add_vline(x=center, annotation_text=name, visible=False, **line_kwargs)

                elif position is not None:
                    fig.add_vline(x=position, annotation_text=name, **line_kwargs)

    def _update_figure(self):
        fig = go.Figure([self._plot,
                         self._avg_plot,
                         self._upper_error_plot,
                         self._lower_error_plot,
                         *self._component_plots])
        fig.update_layout(title=self.title,
                          xaxis_title=self.xaxis_title,
                          yaxis_title=self.yaxis_title)
        if self._invert_spectra_axis:
            fig.update_xaxes(autorange="reversed")
        fig.add_shape(self._energy_line)

        self._add_annotations(fig)

        return fig

    def _update_average_plot(self, selected_data):
        if selected_data is not None and len(selected_data['points']) > 0:
            raveled_indexes = list(map(lambda point: point['pointIndex'], selected_data['points']))
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
        e = click_data["points"][0]["x"]
        self._energy_line.x0 = e
        self._energy_line.x1 = e
        return self._update_figure()

    def _id(self):
        return {'type': 'spectraplot',
                'index': self._parent._instance_index}