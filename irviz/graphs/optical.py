import numpy as np
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
from dash import dcc
from plotly import graph_objects as go

from irviz.components import ColorScaleSelector
from irviz.graphs.slice import SliceGraph
from ryujin.components import Panel
from ryujin.utils.dash import targeted_callback

__all__ = ['OpticalGraph']


class OpticalGraphPanel(Panel):
    def __init__(self, instance_index, cluster_labels):
        self.visibility_toggle = dbc.Checkbox(id=dict(type='optical-visibility', instance_index=instance_index),
                                              checked=True)
        self.color_scale_selector = ColorScaleSelector(_id={'type': 'color-scale-selector',
                                                            'subtype': 'optical',
                                                            'index': instance_index},
                                                       )
        self._cluster_overlay_opacity = dcc.Slider(id={'type': 'cluster-opacity',
                                                       'index': instance_index,
                                                       'subtype': 'optical'},
                                                   min=0,
                                                   max=1,
                                                   step=.05,
                                                   value=.3,
                                                   className='centered-slider',
                                                   disabled=True if cluster_labels is None else False,
                                                   )

        children = [dbc.FormGroup([self.visibility_toggle,
                                   dbc.Label('Show Optical Image')]),
                    dbc.FormGroup([dbc.Label('Color Theme'),
                                   self.color_scale_selector]),
                    dbc.FormGroup(
                        [dbc.Label("Cluster Label Overlay Opacity"), self._cluster_overlay_opacity])]

        super(OpticalGraphPanel, self).__init__('Optical Image', children)

    def init_callbacks(self, app):
        super(OpticalGraphPanel, self).init_callbacks(app)
        self.color_scale_selector.init_callbacks(app)


class OpticalGraph(SliceGraph):
    """Dash Graph for viewing 2D slices of 3D data.

    Parameters
    ----------
    data : dask.array
        3D data array
    parent : html.Div
        The parent object that creates this Graph

    """
    title = 'Optical Image'

    def __init__(self, map_data, instance_index, optical_data, bounds, cluster_labels, cluster_label_names, slice_axis=0,
                 traces=None, shapes=None, **kwargs):
        self.configuration_panel = OpticalGraphPanel(instance_index, cluster_labels)
        self._map_data = map_data

        # map_data is passed here rather than optical_data as a matter of convenience in class composition
        # All the other traces are scaled according to map_data's shape; the image trace is swapped out next
        super(OpticalGraph, self).__init__(map_data,
                                           instance_index,
                                           cluster_labels,
                                           cluster_label_names,
                                           bounds=bounds,
                                           slice_axis=slice_axis,
                                           traces=traces,
                                           shapes=shapes,
                                           **kwargs
                                           # config={'modeBarButtonsToAdd': ['lasso2d']}
                                           )

        # wipe out the free map_data image and replace it with optical_data
        hovertemplate = self._image.hovertemplate
        text = self._image.text
        self._traces.remove(self._image)
        self._image = self._get_image_trace(optical_data,
                                            bounds,
                                            hovertemplate=hovertemplate,
                                            text=text)

        # disable color themes for rgb images
        if isinstance(self._image, go.Image):
            self.configuration_panel.color_scale_selector.disabled = True

        self._traces.insert(0, self._image)
        self.figure = self._update_figure()

    def init_callbacks(self, app):
        super(OpticalGraph, self).init_callbacks(app)

        # Wire-up visibility toggle
        targeted_callback(self._set_visibility,
                          Input(self.configuration_panel.visibility_toggle.id, 'checked'),
                          Output(self.id, 'style'),
                          app=app)

        # Change color scale from selector
        targeted_callback(self.set_color_scale,
                          Input(self.configuration_panel.color_scale_selector.id, 'label'),
                          Output(self.id, 'figure'),
                          app=app)

        self.configuration_panel.init_callbacks(app)

        # update cluster overlay opacity
        targeted_callback(self.update_opacity,
                          Input(self.configuration_panel._cluster_overlay_opacity.id, 'value'),
                          Output(self.id, 'figure'),
                          app=app)

    def _id(self, instance_index):
        _id = super(OpticalGraph, self)._id(instance_index)
        _id['subtype'] = 'optical'
        return _id

    def _show_selection_mask(self, selection):
        if selection is not None:
            # Get x,y from the raveled indexes
            raveled_indexes = list(map(lambda point: point['pointIndex'],
                                       filter(lambda point: point['curveNumber'] == 0,
                                              selection['points'])))
            mask = np.zeros(self._map_data[0].shape)
            # Cannot be 0s - must be NaNs (eval to None) so it doesn't affect underlying HeatMap
            mask.fill(np.NaN)
            mask.ravel()[raveled_indexes] = 1
            # Create overlay
            self._selection_mask.z = mask

        else:
            self._selection_mask.z = np.ones(self._data[0].shape) * np.NaN

        return self._update_figure()
