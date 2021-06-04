from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
import numpy as np
from dash.exceptions import PreventUpdate
from itertools import count

# TODO: implement orthogonal views by using slice_axis kwarg


class SpectraPlotGraph(dcc.Graph):
    _counter = count(0)

    def __init__(self, data, parent):
        self._instance_index = next(self._counter)
        self._data = data
        self._parent = parent

        # Define starting point for energy index (for the slicer line trace)
        default_energy_index = (self._data.shape[0] - 1) // 2

        # Cache the x,y coordinates for slicing into the data for the plot
        #  (default to middle x,y)
        y_index = (self._data.shape[1] - 1) // 2
        x_index = (self._data.shape[2] - 1) // 2

        y = np.asarray(self._data[:, y_index, x_index])
        x = np.arange(0, self._data.shape[0])
        self._plot = go.Scatter(x=x, y=y)

        # x coords positioned relative to the x-axis values
        # y coords positioned according to the plot height (0 = bottom, 1.0 = top)
        self._energy_line = go.layout.Shape(type='line',
                                       # width=3,
                                      xref='x',
                                      yref='paper',
                                       x0=default_energy_index,
                                       x1=default_energy_index,
                                       y0=0,
                                       y1=1)

        fig = self._update_figure()

        super(SpectraPlotGraph, self).__init__(id=self._id(),
                                               figure=fig)

    def register_callbacks(self):
        self._parent._app.callback(
            Output(self.id, 'figure'),
            Input(self._parent.slice_graph.id, 'clickData'),
            Input(self.id, 'clickData')
        )(self._show_plot)

    def _update_figure(self):
        fig = go.Figure(self._plot)
        fig.update_layout(title=f'Spectra Intensities',
                          xaxis_title="Spectra",
                          yaxis_title="Intensity")
        fig.add_shape(self._energy_line)
        return fig

    def _show_plot(self, slice_graph_click_data, self_click_data):
        if slice_graph_click_data is None and self_click_data is None:
            raise PreventUpdate

        # When the slice graph is clicked, update plot with the clicked x,y coord
        if slice_graph_click_data is not None:
            y_index = slice_graph_click_data["points"][0]["y"]
            x_index = slice_graph_click_data["points"][0]["x"]
            self._plot.y = np.asarray(self._data[:, y_index, x_index])
            self._plot.x = np.arange(0, self._data.shape[0])

        # When this SpectraGraph itself is clicked, update the energy slicer line
        if self_click_data is not None:
            energy_index = self_click_data["points"][0]["x"]
            self._energy_line.x0 = energy_index
            self._energy_line.x1 = energy_index

        return self._update_figure()

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

    def __init__(self, data, parent, slice_axis=0):

        # Cache our data and parent for use in the callbacks
        self._data = data
        self._parent = parent
        self._instance_index = next(self._counter)

        default_energy_index = (self._data.shape[0] - 1) // 2

        # Create traces (i.e. 'glyphs') that will comprise a plotly Figure
        self._image = go.Heatmap(z=np.asarray(self._data[default_energy_index]))
        self._h_line = go.layout.Shape(type='line',
                                       # width=3,
                                       xref='paper',
                                       yref='y',
                                       x0=0,
                                       x1=1,
                                       y0=(self._data.shape[1] - 1) // 2,
                                       y1=(self._data.shape[1] - 1) // 2)
        self._v_line = go.layout.Shape(type='line',
                                       xref='x',
                                       yref='paper',
                                       x0=(self._data.shape[2] - 1) // 2,
                                       x1=(self._data.shape[2] - 1) // 2,
                                       y0=0,
                                       y1=1)

        figure = self._update_figure()
        super(SliceGraph, self).__init__(figure=figure,
                                         id=self._id())

    def _id(self):
        return f'slicegraph_{self._instance_index}'

    def register_callbacks(self):
        # Set up callbacks
        # ----------------

        # When the parent viewer's 'spectra_graph' is clicked
        #     we need to update the internal Figure for this Graph

        print('ids:', self.id, self._parent.spectra_graph.id)

        self._parent._app.callback(
            Output(self.id, 'figure'),
            Input(self._parent.spectra_graph.id, 'clickData'),
            Input(self.id, 'clickData')
        )(self.show_slice)

    def _update_figure(self):
        """ Remake the figure to force a display update """
        figure = go.Figure([self._image])
        figure.add_shape(self._h_line)
        figure.add_shape(self._v_line)
        return figure

    def show_slice(self, spectra_graph_click_data, self_click_data):
        """Show a 2D slice at a specific energy defined in click data.

        Parameters
        ----------
        spectra_graph_click_data : dict
            Dictionary that contains point info from where the input Graph was clicked.
        """
        if spectra_graph_click_data is None and self_click_data is None:
            raise PreventUpdate

        # When the spectra graph is clicked, update image slicing
        if spectra_graph_click_data is not None:
            slice_index = spectra_graph_click_data["points"][0]["x"]
            self._image.z = np.asarray(self._data[slice_index])

        # When this SliceGraph itself is clicked, update its x,y slicer lines
        if self_click_data:
            y_index = self_click_data["points"][0]["y"]
            x_index = self_click_data["points"][0]["x"]
            self._h_line.y0 = y_index
            self._h_line.y1 = y_index
            self._v_line.x0 = x_index
            self._v_line.x1 = x_index

        # Need to update our figure again when we update the traces
        return self._update_figure()


class PairPlotGraph(dcc.Graph):
    _counter = count(0)

    def __init__(self, data, parent):
        self._instance_index = next(self._counter)

        # Cache our data and parent for use in the callbacks
        self._data = data
        self._parent = parent

        # Create traces (i.e. 'glyphs') that will comprise a plotly Figure
        self._scatter = go.Scatter(x=[], y=[], mode='markers+text')

        figure = self._update_figure()
        super(PairPlotGraph, self).__init__(figure=figure,
                                         id=f'pair_plot_{self._instance_index}')

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

    def _show_selection_info(self, selected_data):
        if not selected_data:
            return "info"
        return str(list(map(lambda point: point['pointIndex'], selected_data['points'])))

    def _update_figure(self):
        """ Remake the figure to force a display update """
        figure = go.Figure([self._scatter])
        return figure

    def show_pair_plot(self, component1, component2):
        if component1 is None and component2 is None:
            raise PreventUpdate

        self._scatter.x = np.asarray(self._data[component1].ravel())
        self._scatter.y = np.asarray(self._data[component2].ravel())

        return self._update_figure()


class DecompositionGraph(SliceGraph):

    def register_callbacks(self):
        # Set up callbacks
        # ----------------

        # When the parent viewer's 'spectra_graph' is clicked
        #     we need to update the internal Figure for this Graph
        print('ids:', self.id, self._parent.decomposition_component_selector.id)
        self._parent._app.callback(
            Output(self.id, 'figure'),
            Input(self.id, 'clickData'),
            Input(self._parent.decomposition_component_selector.id, 'value')
        )(self.show_slice)

    def show_slice(self, self_click_data, component_index):
        """Show a 2D slice at a specific energy defined in click data.

        Parameters
        ----------
        spectra_graph_click_data : dict
            Dictionary that contains point info from where the input Graph was clicked.
        """

        print('test')
        if component_index is None and self_click_data is None:
            raise PreventUpdate

            # When the spectra graph is clicked, update image slicing
        if component_index is not None:
            self._image.z = np.asarray(self._data[component_index])

        # When this SliceGraph itself is clicked, update its x,y slicer lines
        if self_click_data:
            y_index = self_click_data["points"][0]["y"]
            x_index = self_click_data["points"][0]["x"]
            self._h_line.y0 = y_index
            self._h_line.y1 = y_index
            self._v_line.x0 = x_index
            self._v_line.x1 = x_index

            # Need to update our figure again when we update the traces
        return self._update_figure()
