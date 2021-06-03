import numpy
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
import numpy as np
from dash.exceptions import PreventUpdate


class SliceGraph(dcc.Graph):
    """Dash Graph for viewing 2D slices of 3D data.

    Parameters
    ----------
    data : dask.array
        3D data array
    parent : dash_html_components.Div
        The parent object that creates this Graph

    """
    _counter = 0

    def __init__(self, data, parent):
        SliceGraph._counter += 1

        # Cache our data and parent for use in the callbacks
        self._data = data
        self._parent = parent

        # Create traces (i.e. 'glyphs') that will comprise a plotly Figure
        self._image = go.Heatmap(z=np.asarray(self._data[0]))
        self._h_line = go.layout.Shape(type='line',
                                       # width=3,
                                       x0=0,
                                       x1=self._data.shape[2] - 1,
                                       y0=self._data.shape[1] / 2,
                                       y1=self._data.shape[1] / 2)

        figure = self._update_figure()
        super(SliceGraph, self).__init__(figure=figure,
                                         id=f'sliceview_{self._counter}')

        # Set up callbacks
        # ----------------

        # When the parent viewer's 'spectra_graph' is clicked
        #     we need to update the internal Figure for this Graph
        self._parent._app.callback(
            Output(self.id, 'figure'),
            Input(self._parent.spectra_graph.id, 'clickData')
        )(self.show_slice)

    def _update_figure(self):
        figure = go.Figure([self._image])
        figure.add_shape(self._h_line)
        return figure

    def show_slice(self, click_data):
        """Show a 2D slice at a specific energy defined in click data.

        Parameters
        ----------
        click_data : dict
            Dictionary that contains point info from where the input Graph was clicked.
        """
        if click_data is None:
            raise PreventUpdate

        energy_index = click_data["points"][0]["x"]
        self._image.z = numpy.asarray(self._data[energy_index])
        # Need to update our figure again when we update the traces
        return self._update_figure()

    def show_location(self, location):
        ...
