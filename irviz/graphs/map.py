from dash.dependencies import Input, Output

from irviz.graphs.slice import SliceGraph
from irviz.utils.dash import targeted_callback
__all__ = ['MapGraph']


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

    def register_callbacks(self):
        super(MapGraph, self).register_callbacks()

        # When the spectra graph is clicked, update image slicing
        targeted_callback(self.update_slice,
                          Input(self._parent.spectra_graph.id, 'clickData'),
                          Output(self.id, 'figure'),
                          app=self._parent._app)

        # Change color scale from selector
        targeted_callback(self.set_color_scale,
                          Input(self._parent.map_color_scale_selector.id, 'label'),
                          Output(self.id, 'figure'),
                          app=self._parent._app)

    def _id(self):
        _id = super(MapGraph, self)._id()
        _id['subtype'] = 'map'
        return _id

    @property
    def map(self):
        """The currently displayed map slice at the current spectral index"""
        return self._image.z
