import dash
import numpy as np
from dash._utils import create_callback_id
from dash.dependencies import Input, Output, State
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
                                'selected_rows'),
                          Output(self.id, 'figure'),
                          State(dict(type='parameter-set-selector',
                                     index=self._instance_index),
                                'data'),
                          app=app)

        super(BackgroundMapGraph, self).init_callbacks(app)

    def update_selection(self, selected_rows):
        _id = create_callback_id(State(dict(type='parameter-set-selector',
                                            index=self._instance_index), 'data'))
        parameter_set_list = dash.callback_context.states[_id] or []

        if selected_rows and selected_rows[0] <= len(parameter_set_list) - 1:
            mask = np.asarray(parameter_set_list[selected_rows[0]]['map_mask']).astype('O')
            if mask is None:
                np.ones_like(self._image.z) * np.NaN
            else:
                mask[np.logical_not(mask.astype(np.bool_))] = np.NaN
                mask[mask == 1] = 1  # casts True -> 1
            self._selection_mask.z = mask
        else:
            self._selection_mask.z = np.ones_like(self._image.z) * np.NaN

        return self._update_figure()
