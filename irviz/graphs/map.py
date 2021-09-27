from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html

from irviz.components import ColorScaleSelector
from irviz.graphs.slice import SliceGraph
from ryujin.components import Panel
from ryujin.utils.dash import targeted_callback

__all__ = ['MapGraph']


class MapGraphPanel(Panel):
    def __init__(self, instance_index, cluster_labels):
        self.visibility_toggle = dbc.Checkbox(id=dict(type='map-visibility', instance_index=instance_index), checked=True)
        self._map_color_scale_selector = ColorScaleSelector(_id='map-color-scale-selector',
                                                            value='Viridis')
        self._cluster_overlay_opacity = dcc.Slider(id={'type': 'cluster-opacity',
                                                       'index': instance_index,
                                                       'subtype': 'map'},
                                                   min=0,
                                                   max=1,
                                                   step=.05,
                                                   value=.3,
                                                   className='centered-slider',
                                                   disabled=True if cluster_labels is None else False,
                                                   )

        map_settings_form = dbc.Form([dbc.FormGroup([dbc.Label("Map Color Scale"), self._map_color_scale_selector]),
                                      dbc.FormGroup(
                                          [dbc.Label("Cluster Label Overlay Opacity"), self._cluster_overlay_opacity])])

        children = [dbc.FormGroup([self.visibility_toggle,
                                   dbc.Label('Show Map Image')]),
                    map_settings_form]

        super(MapGraphPanel, self).__init__('Map Image', children)

    def init_callbacks(self, app):
        super(MapGraphPanel, self).init_callbacks(app)
        self._map_color_scale_selector.init_callbacks(app)


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

    def __init__(self, *args, **kwargs):
        super(MapGraph, self).__init__(*args, **kwargs)
        self.configuration_panel = MapGraphPanel(self._instance_index, self._cluster_labels)

    def init_callbacks(self, app):
        super(MapGraph, self).init_callbacks(app)

        # Wire-up visibility toggle
        targeted_callback(self._set_visibility,
                          Input(self.configuration_panel.visibility_toggle.id, 'checked'),
                          Output(self.id, 'style'),
                          app=app)

        # When the spectra graph is clicked, update image slicing
        targeted_callback(self.update_slice,
                          Input({'type': 'spectraplot',
                                 'index': self._instance_index}, 'clickData'),
                          Output(self.id, 'figure'),
                          app=app)

        # Change color scale from selector
        targeted_callback(self.set_color_scale,
                          Input(self.configuration_panel._map_color_scale_selector.id, 'label'),
                          Output(self.id, 'figure'),
                          app=app)

        self.configuration_panel.init_callbacks(app)

    def _id(self, instance_index):
        _id = super(MapGraph, self)._id(instance_index)
        _id['subtype'] = 'map'
        return _id

    @property
    def map(self):
        """The currently displayed map slice at the current spectral index"""
        return self._image.z
