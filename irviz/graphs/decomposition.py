import re

import dash
import numpy as np
from dash.dependencies import Output, Input, ALL
from plotly import graph_objects as go
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc

from irviz.components import ColorScaleSelector
from irviz.graphs._colors import decomposition_color_scales, transparent_color_scales
from irviz.graphs.slice import SliceGraph
from ryujin.components import Panel
from ryujin.utils.dash import targeted_callback

__all__ = ['DecompositionGraph']


class DecompositionGraphPanel(Panel):
    def __init__(self, instance_index, component_count, cluster_labels):
        self._cluster_overlay_opacity = dcc.Slider(id={'type': 'cluster-opacity',
                                                       'index': instance_index,
                                                       'subtype': 'decomposition'},
                                                   min=0,
                                                   max=1,
                                                   step=.05,
                                                   value=.3,
                                                   className='centered-slider',
                                                   disabled=True if cluster_labels is None else False,
                                                   )

        radio_kwargs = dict(className='btn-group-vertical col-sm-auto',
                            labelClassName="btn btn-secondary",
                            labelCheckedClassName="active",
                            options=[{'label': f'{i + 1}', 'value': i}
                                     for i in range(component_count)]

                            )

        self._decomposition_component_selector = dbc.Checklist(id='decomposition-component-selector',
                                                               value=[0],
                                                               style={'paddingLeft': 0, 'paddingRight': 0},
                                                               **radio_kwargs)

        self._component_opacity_sliders = html.Div(
            [dcc.Slider(
                id={'type': 'component-opacity',
                    'index': i},
                min=0,
                max=1,
                step=.1,
                value=.5 if i else 1,
                className='centered-slider',
                disabled=True if i else False
            ) for i in range(component_count)],
            className='col-sm',
            style={'paddingLeft': 0, 'paddingRight': 0},
            id='component-opacity-sliders'
        )

        self.color_scale_selectors = [ColorScaleSelector({'type': 'color-scale-selector',
                                                          'index': i},
                                                         values=decomposition_color_scales,
                                                         value=decomposition_color_scales[
                                                             i % len(decomposition_color_scales)],
                                                         )
                                      for i in range(component_count)]

        self._component_color_scale_selectors = html.Div(self.color_scale_selectors,
                                                         className='col-sm-auto',
                                                         style={'paddingLeft': 0, 'paddingRight': 0, 'marginTop': 2.5},
                                                         )

        decomposition_selector_layout = dbc.FormGroup(
            [
                html.Div([
                    html.Div([self._decomposition_component_selector,
                              self._component_color_scale_selectors,
                              self._component_opacity_sliders, ],
                             className='row well'
                             ),
                ],
                    className='container'
                ),
            ],
            className='radio-group'
        )

        self.visibility_toggle = dbc.Checkbox(id=dict(type='decomposition-visibility',
                                                      instance_index=instance_index),
                                              checked=True)

        children = [dbc.FormGroup([self.visibility_toggle,
                                   dbc.Label('Show Decomposition Image')]),
                    dbc.FormGroup([dbc.Label('Component Color Themes'),
                                   decomposition_selector_layout]),
                    dbc.FormGroup(
                        [dbc.Label("Cluster Label Overlay Opacity"), self._cluster_overlay_opacity])]

        super(DecompositionGraphPanel, self).__init__('Decomposition Image', children)

    def init_callbacks(self, app):
        super(DecompositionGraphPanel, self).init_callbacks(app)
        for color_scale_selector in self.color_scale_selectors:
            color_scale_selector.init_callbacks(app)

        # Disable sliders when their component is hidden
        targeted_callback(self.disable_sliders,
                          Input(self._decomposition_component_selector.id, 'value'),
                          Output(self._component_opacity_sliders.id, 'children'),
                          app=app)

    def disable_sliders(self, component_indices):
        for i, slider in enumerate(self._component_opacity_sliders.children):
            slider.disabled = not (i in component_indices)

        return self._component_opacity_sliders.children


class DecompositionGraph(SliceGraph):
    title = 'Decomposition Maps'

    def __init__(self, data, instance_index, cluster_labels, cluster_label_names, bounds, *args, **kwargs):
        self.configuration_panel = DecompositionGraphPanel(instance_index, data.shape[0], cluster_labels)

        self._component_traces = []
        for i in range(data.shape[0]):
            color_scale = decomposition_color_scales[i % len(decomposition_color_scales)]
            color_scale = transparent_color_scales.get(color_scale, color_scale)

            self._component_traces.append(go.Heatmap(z=np.asarray(data[i]),
                                                     colorscale=color_scale,
                                                     y0=bounds[1][0],
                                                     dy=(bounds[1][1] - bounds[1][0]) / (data.shape[1] - 1),
                                                     x0=bounds[2][0],
                                                     dx=(bounds[2][1] - bounds[2][0]) / (data.shape[2] - 1),
                                                     visible=(i == 0),
                                                     opacity=.5 if i else 1,
                                                     ))

        kwargs['traces'] = self._component_traces

        super(DecompositionGraph, self).__init__(data,
                                                 instance_index,
                                                 cluster_labels,
                                                 cluster_label_names,
                                                 bounds,
                                                 *args,
                                                 **kwargs)

        # Hide the free image trace
        self._image.visible = False
        self.figure = self._update_figure()

    def _id(self, instance_index):
        _id = super(DecompositionGraph, self)._id(instance_index)
        _id['subtype'] = 'decomposition'
        return _id

    def init_callbacks(self, app):
        super(DecompositionGraph, self).init_callbacks(app)

        # Wire-up visibility toggle
        targeted_callback(self._set_visibility,
                          Input(self.configuration_panel.visibility_toggle.id, 'checked'),
                          Output(self.id, 'style'),
                          app=app)

        # Wire-up opacity sliders
        targeted_callback(self.set_component_opacity,
                          Input({'type': 'component-opacity', 'index': ALL}, 'value'),
                          Output(self.id, 'figure'),
                          app=app)

        # Show components when selected
        targeted_callback(self.show_components,
                          Input(self.configuration_panel._decomposition_component_selector.id, 'value'),
                          Output(self.id, 'figure'),
                          app=app)

        # Update the color scale when new item is selected
        targeted_callback(self.set_color_scale,
                          Input({'type': 'color-scale-selector', 'index': ALL}, 'label'),
                          Output(self.id, 'figure'),
                          app=app)

        self.configuration_panel.init_callbacks(app)

    def set_color_scale(self, color_scale):
        i = int(re.findall('(?<="index":)\\d+(?=,)', dash.callback_context.triggered[0]['prop_id'])[0])
        color_scale = transparent_color_scales.get(color_scale, color_scale)
        self._component_traces[i].colorscale = color_scale

        return self._update_figure()

    def _opacity_slider(self, i):
        return self.configuration_panel._component_opacity_sliders.children[i]

    def _color_scale_selector(self, i):
        return self.configuration_panel._component_color_scale_selectors.children[i]

    def set_component_opacity(self, value):
        i = int(re.findall('(?<="index":)\\d+(?=,)', dash.callback_context.triggered[0]['prop_id'])[0])
        self._opacity_slider(i).value = value
        self._update_opacity()
        return self._update_figure()

    def _update_opacity(self):
        # Get a sum of all enabled slider values minus the first enabled value
        total = 0
        for slider in self.configuration_panel._component_opacity_sliders.children:
            if not slider.disabled:
                total += slider.value

        # Set each trace's opacity to a value proportional to its weight; always set first visible trace's opacity to 1
        bg_set = False
        for i, trace in enumerate(self._component_traces):
            if trace.visible:
                if not bg_set:
                    trace.opacity = 1
                    bg_set = True
                    continue

                trace.opacity = self._opacity_slider(i).value / total

    def show_components(self, component_indices):
        for i, trace in enumerate(self._component_traces):
            trace.visible = (i in component_indices)
            trace.showscale = len(component_indices) < 2
        self._update_opacity()

        return self._update_figure()
