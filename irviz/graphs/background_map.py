import numpy as np
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html

from irviz.components import ColorScaleSelector
from irviz.graphs.map import MapGraphPanel, MapGraph
from irviz.graphs.slice import SliceGraph
from ryujin.components import Panel
from ryujin.utils.dash import targeted_callback

__all__ = ['BackgroundMapGraph']


class BackgroundMapGraph(MapGraph):
    """Dash Graph for viewing 2D slices of 3D data and selecting regions for background removal

    Parameters
    ----------
    data : dask.array
        3D data array
    parent : html.Div
        The parent object that creates this Graph

    """

    def init_callbacks(self, app):
        # update selection when active parameter set changes
        targeted_callback(self.update_selection,
                          Input(dict(type='parameter-set-selector',
                                     index=self._instance_index),
                                'data'),
                          Output(self.id, 'figure'),
                          app=app)

        super(BackgroundMapGraph, self).init_callbacks(app)

    def update_selection(self, data):
        self._selection_mask.z = next(iter(filter(lambda record: record.get('selected'), data)))['map_mask'] or np.ones_like(self._image.z) * np.NaN

        return self._update_figure()
