from itertools import count

import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import plotly.graph_objects as go
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate

from dash.development.base_component import Component

# TODO: implement orthogonal views by using slice_axis kwarg


def nearest_bin(x, bounds, bin_count):
    return int((x-bounds[0])/(bounds[1]-bounds[0])*bin_count)


class SpectraPlotGraph(dcc.Graph):
    _counter = count(0)

    title = 'Spectra Intensities'
    xaxis_title = 'Spectral Unit'
    yaxis_title = 'Intensity'

    def __init__(self, data, bounds, parent, labels=None):
        """Interactive Graph that shows spectral intensities at a selectable energy / wave-number index.

        Parameters
        ----------
        data : dask array
            3D array containing data with axes E (or wave-number), y, and x for displaying in the Graph
        bounds : ndarray-like
            3D array defining the bounds (min & max) pairs for E (or wave-number), y, and x data
        parent : Component
            Reference to Component that created this Graph (for registering callbacks)
        labels : dict[str, str]
            Optional dictionary with keys `xaxis_title`, `yaxis_title`, and `title` that define the Graph's labels
        """
        self._instance_index = next(self._counter)
        self._data = data
        self._parent = parent
        self._bounds = bounds
        labels = labels or dict()
        self.xaxis_title = labels.get('xaxis_title', self.xaxis_title)
        self.yaxis_title = labels.get('yaxis_title', self.yaxis_title)
        self.title = labels.get('title', self.title)

        # Cache the x,y coordinates (from map graph) for slicing into the data for the plot
        #  (default to middle x,y)
        self._y_index = (self._data.shape[1] - 1) // 2
        self._x_index = (self._data.shape[2] - 1) // 2

        y = np.asarray(self._data[:, self._y_index, self._x_index])
        x = np.linspace(bounds[0][0], bounds[0][1], self._data.shape[0])
        self._plot = go.Scatter(x=x, y=y)

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

        fig = self._update_figure()

        super(SpectraPlotGraph, self).__init__(id=self._id(),
                                               figure=fig,
                                               className='col-lg-12')

    def register_callbacks(self):
        self._parent._app.callback(
            Output(self.id, 'figure'),
            Input(self._parent.map_graph.id, 'clickData'),
            Input(self.id, 'clickData'),
            Input(self._parent.decomposition_graph.id, 'clickData')
        )(self._show_plot)

        # Wire-up visibility toggle
        self._parent._app.callback(
            Output(self.id, 'style'),
            Input(self._parent.graph_toggles.id, 'value')
        )(self._set_visibility)
        
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
            return {'display':'block'}
        else:
            return {'display':'none'}

    def _update_figure(self):
        fig = go.Figure(self._plot)
        fig.update_layout(title=self.title,
                          xaxis_title=self.xaxis_title,
                          yaxis_title=self.yaxis_title)
        fig.add_shape(self._energy_line)
        return fig

    def _show_plot(self, slice_graph_click_data, self_click_data, decomposition_click_data):
        triggered = dash.callback_context.triggered
        if not triggered:
            raise PreventUpdate

        # When the slice graph is clicked, update plot with the clicked x,y coord
        if self._parent.map_graph.id in triggered[0]['prop_id']:
            self._show_click(slice_graph_click_data)

        elif self._parent.decomposition_graph.id in triggered[0]['prop_id']:
            self._show_click(decomposition_click_data)

        # When this SpectraGraph itself is clicked, update the energy slicer line
        elif self_click_data is not None:
            e = self_click_data["points"][0]["x"]
            self._energy_line.x0 = e
            self._energy_line.x1 = e

        return self._update_figure()

    def _show_click(self, click_data):
        y = click_data["points"][0]["y"]
        x = click_data["points"][0]["x"]
        self._x_index = nearest_bin(x, self._bounds[2], self._data.shape[2])
        self._y_index = nearest_bin(y, self._bounds[1], self._data.shape[1])
        self._plot.y = np.asarray(self._data[:, self._y_index, self._x_index])

    def _id(self):
        return f'spectraplot_{self._instance_index}'


class SliceGraph(dcc.Graph):
    """Dash Graph for viewing 2D slices of 3D data.

    Parameters
    ----------
    data : dask.array
        3D data array
    parent : html.Div
        The parent object that creates this Graph

    """
    _counter = count(0)

    title = ''
    xaxis_title = 'X'
    yaxis_title = 'Y'
    aspect_locked = True

    def __init__(self, data, bounds, parent, slice_axis=0, traces=None, shapes=None):

        # Cache our data and parent for use in the callbacks
        self._data = data
        self._parent = parent
        self._bounds = bounds
        self._instance_index = next(self._counter)
        self._traces = traces or []
        self._shapes = shapes or []

        default_slice_index = self._init_slice_index()  # TODO: Refactor these classes with a base

        # Create traces (i.e. 'glyphs') that will comprise a plotly Figure
        self._image = go.Heatmap(z=np.asarray(self._data[default_slice_index]),
                                 colorscale='gray',
                                 y0=bounds[1][0],
                                 dy=(bounds[1][1]-bounds[1][0])/data.shape[1],
                                 x0=bounds[2][0],
                                 dx=(bounds[2][1]-bounds[2][0])/data.shape[2],
                                 )

        self._h_line = go.layout.Shape(type='line',
                                       # width=3,
                                       xref='paper',
                                       yref='y',
                                       x0=0,
                                       x1=1,
                                       y0=(bounds[1][1]+bounds[1][0])/2,
                                       y1=(bounds[1][1]+bounds[1][0])/2)
        self._v_line = go.layout.Shape(type='line',
                                       xref='x',
                                       yref='paper',
                                       x0=(bounds[2][1]+bounds[2][0])/2,
                                       x1=(bounds[2][1]+bounds[2][0])/2,
                                       y0=0,
                                       y1=1)

        self._traces.insert(0, self._image)
        self._shapes.extend([self._h_line, self._v_line])

        figure = self._update_figure()
        super(SliceGraph, self).__init__(figure=figure,
                                         id=self._id(),
                                         className='col-lg-4')

    def _id(self):
        return f'slicegraph_{self._instance_index}'

    def _init_slice_index(self):
        ...

    def register_callbacks(self):
        ...

    def _update_figure(self):
        fig = go.Figure(self._traces)
        fig.update_layout(title=self.title,
                          xaxis_title=self.xaxis_title,
                          yaxis_title=self.yaxis_title)
        if self.aspect_locked:
            fig.update_yaxes(scaleanchor="x", scaleratio=1)
        for shape in self._shapes:
            fig.add_shape(shape)
        return fig

    def _show_click(self, click_data):
        y_index = click_data["points"][0]["y"]
        x_index = click_data["points"][0]["x"]
        self._h_line.y0 = y_index
        self._h_line.y1 = y_index
        self._v_line.x0 = x_index
        self._v_line.x1 = x_index
        
    @property
    def position(self):
        """The current spatial position of the crosshair"""
        return self._v_line.x0, self._h_line.y0


class MapGraph(SliceGraph):
    """Dash Graph for viewing 2D slices of 3D data.

    Parameters
    ----------
    data : dask.array
        3D data array
    parent : html.Div
        The parent object that creates this Graph

    """
    title = 'IR Spectral Map'

    def __init__(self, data, bounds, parent, slice_axis=0, traces=None, shapes=None):
        self._selection_mask = go.Heatmap(z=np.ones(data[0].shape) * np.NaN,
                                          colorscale='reds',
                                          opacity=0.3,
                                          showscale=False,
                                          y0=bounds[1][0],
                                          dy=(bounds[1][1]-bounds[1][0])/data.shape[1],
                                          x0=bounds[2][0],
                                          dx=(bounds[2][1]-bounds[2][0])/data.shape[2])
        traces = (traces or []) + [self._selection_mask]
        super(MapGraph, self).__init__(data, bounds, parent, slice_axis=0, traces=traces, shapes=shapes)

    def register_callbacks(self):
        # Set up callbacks
        # ----------------

        # When the parent viewer's 'spectra_graph' is clicked
        #     we need to update the internal Figure for this Graph

        self._parent._app.callback(
            Output(self.id, 'figure'),
            Input(self._parent.spectra_graph.id, 'clickData'),
            Input(self.id, 'clickData'),
            Input(self._parent.decomposition_graph.id, 'clickData'),
            Input(self._parent.pair_plot_graph.id, 'selectedData')
        )(self.show_slice)

    def show_slice(self, spectra_graph_click_data, self_click_data, decomposition_click_data, pair_plot_selection):
        """Show a 2D slice at a specific energy defined in click data.

        Parameters
        ----------
        spectra_graph_click_data : dict
            Dictionary that contains point info from where the input Graph was clicked.
        """
        triggered = dash.callback_context.triggered
        if not triggered:
            raise PreventUpdate

        # When the spectra graph is clicked, update image slicing
        if self._parent.spectra_graph.id in triggered[0]['prop_id']:
            slice = spectra_graph_click_data["points"][0]["x"]
            slice_index = nearest_bin(slice, self._bounds[0], self._data.shape[0])
            self._image.z = np.asarray(self._data[slice_index])

        # When this SliceGraph itself is clicked, update its x,y slicer lines
        if self.id in triggered[0]['prop_id']:
            self._show_click(self_click_data)

        # When the decomposition graph is clicked show the same position
        if self._parent.decomposition_graph.id in triggered[0]['prop_id']:
            self._show_click(decomposition_click_data)

        # When points are selected in the pair plot, show them here
        if self._parent.pair_plot_graph.id in triggered[0]['prop_id']:
            self._show_selection_mask(pair_plot_selection)

        # Need to update our figure again when we update the traces
        return self._update_figure()

    def _show_selection_mask(self, selection):
        # Get x,y from the raveled indexes
        raveled_indexes = list(map(lambda point: point['pointIndex'], selection['points']))
        mask = np.zeros(self._data[0].shape)
        # Cannot be 0s - must be NaNs (eval to None) so it doesn't affect underlying HeatMap
        mask.fill(np.NaN)
        mask.ravel()[raveled_indexes] = 1
        # Create overlay
        self._selection_mask.z = mask

    def _init_slice_index(self):
        return (self._data.shape[0] - 1) // 2


class DecompositionGraph(SliceGraph):
    title = 'Decomposition Maps'

    def register_callbacks(self):
        # Set up callbacks
        # ----------------

        # When the parent viewer's 'spectra_graph' is clicked
        #     we need to update the internal Figure for this Graph
        self._parent._app.callback(
            Output(self.id, 'figure'),
            Input(self.id, 'clickData'),
            Input(self._parent.decomposition_component_selector.id, 'value'),
            Input(self._parent.map_graph.id, 'clickData')
        )(self.show_slice)

        # Wire-up visibility toggle
        self._parent._app.callback(
            Output(self.id, 'style'),
            Input(self._parent.graph_toggles.id, 'value')
        )(self._set_visibility)

    def show_slice(self, self_click_data, component_index, slice_click_data):
        """Show a 2D slice at a specific energy defined in click data.

        Parameters
        ----------
        spectra_graph_click_data : dict
            Dictionary that contains point info from where the input Graph was clicked.
        """
        triggered = dash.callback_context.triggered
        if not triggered:
            raise PreventUpdate

        # When the decomposition_component_selector is clicked, update image slicing
        if self._parent.decomposition_component_selector.id in triggered[0]['prop_id']:
            self._image.z = np.asarray(self._data[component_index])

        # When this DecompositionGraph itself is clicked, update its x,y slicer lines
        if self.id in triggered[0]['prop_id']:
            self._show_click(self_click_data)

        if self._parent.map_graph.id in triggered[0]['prop_id']:
            self._show_click(slice_click_data)

        # Need to update our figure again when we update the traces
        return self._update_figure()

    def _init_slice_index(self):
        return 0

    @staticmethod
    def _set_visibility(switches_value):
        if 'show_decomposition' in switches_value:
            return {'display':'block'}
        else:
            return {'display':'none'}


class PairPlotGraph(dcc.Graph):
    _counter = count(0)

    def __init__(self, data, parent):
        self._instance_index = next(self._counter)

        # Cache our data and parent for use in the callbacks
        self._data = data
        self._parent = parent
        self._component1 = self._component2 = 0

        # Create traces (i.e. 'glyphs') that will comprise a plotly Figure
        self._scatter = go.Scattergl(x=[], y=[], mode='markers')

        figure = self._update_figure()
        super(PairPlotGraph, self).__init__(figure=figure,
                                            id=f'pair_plot_{self._instance_index}',
                                            className='col-lg-4')

    def register_callbacks(self):
        # Set up callbacks
        # ----------------

        # When the parent viewer's 'spectra_graph' is clicked
        #     we need to update the internal Figure for this Graph
        self._parent._app.callback(
            Output(self.id, 'figure'),
            Input(self._parent.decomposition_component_1.id, 'value'),
            Input(self._parent.decomposition_component_2.id, 'value'),
        )(self.show_pair_plot)

        # Set up selection tool callbacks
        self._parent._app.callback(
            Output(self._parent.info_content.id, 'children'),
            Input(self.id, 'selectedData')
        )(self._show_selection_info)

        # Wire-up visibility toggle
        self._parent._app.callback(
            Output(self.id, 'style'),
            Input(self._parent.graph_toggles.id, 'value')
        )(self._set_visibility)

    def _show_selection_info(self, selected_data):
        if not selected_data:
            return "info"
        return str(list(map(lambda point: point['pointIndex'], selected_data['points'])))

    def _update_figure(self):
        """ Remake the figure to force a display update """
        fig = go.Figure([self._scatter])
        fig.update_layout(title=f'Pair Plot',
                          xaxis_title=f'Component #{self._component1+1}',
                          yaxis_title=f'Component #{self._component2+1}')
        return fig

    def show_pair_plot(self, component1, component2):
        if component1 is None and component2 is None:
            raise PreventUpdate

        x = self._data[component1]
        y = self._data[component2]
        self._scatter.x = np.asarray(x.ravel())
        self._scatter.y = np.asarray(y.ravel())
        self._component1 = component1
        self._component2 = component2

        return self._update_figure()

    @staticmethod
    def _set_visibility(switches_value):
        if 'show_pair_plot' in switches_value:
            return {'display':'block'}
        else:
            return {'display':'none'}
