import re

import dash
import numpy as np
from dash._utils import stringify_id
from dash.dependencies import Output, Input, ALL, MATCH, State
from dash.exceptions import PreventUpdate
from plotly import graph_objects as go
import dash_bootstrap_components as dbc
from dash import html
from dash import dcc

from irviz.components import ColorScaleSelector
from irviz.graphs._colors import decomposition_color_scales, transparent_color_scales
from irviz.graphs.slice import SliceGraph
from ryujin.components import Panel
from ryujin.utils.dash import targeted_callback

__all__ = ['DecompositionGraph']


class DecompositionGraphPanel(Panel):
    def __init__(self, instance_index, component_count, cluster_labels):
        self._checkboxes = []
        self._check_labels = []
        self._component_color_scale_selectors = []
        self._component_opacity_sliders = []

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
        items = [self.build_item(i) for i in range(component_count)]

        self.decomposition_selector_layout = dbc.FormGroup(items, id='decomposition-selector-layout')

        self.visibility_toggle = dbc.Checkbox(id=dict(type='decomposition-visibility',
                                                      instance_index=instance_index),
                                              checked=True)

        children = [dbc.FormGroup([self.visibility_toggle,
                                   dbc.Label('Show Decomposition Image')]),
                    dbc.FormGroup([dbc.Label('Component Color Themes'),
                                   self.decomposition_selector_layout]),
                    dbc.FormGroup(
                        [dbc.Label("Cluster Label Overlay Opacity"), self._cluster_overlay_opacity])]

        super(DecompositionGraphPanel, self).__init__('Decomposition Image', children)

    def init_callbacks(self, app):
        super(DecompositionGraphPanel, self).init_callbacks(app)
        for color_scale_selector in self._component_color_scale_selectors:
            color_scale_selector.init_callbacks(app)

        # Disable sliders when their component is hidden
        targeted_callback(self.disable_slider,
                          Input(dict(type='decomposition-component-toggle', index=MATCH), 'checked'),
                          Output(dict(type='component-opacity', index=MATCH), 'disabled'),
                          app=app)

        # Change number of items when decomposition changes
        targeted_callback(self.update_items,
                          Input(dict(type='decomposition-status', pattern=ALL), 'data'),
                          Output(self.decomposition_selector_layout.id, 'children'),
                          app=app)

        # propagate check state to label via 'active' class name
        targeted_callback(self.update_check_state,
                          Input(dict(type='decomposition-component-toggle', index=MATCH), 'checked'),
                          Output(dict(type='decomposition-component-label', index=MATCH), 'className'),
                          State(dict(type='decomposition-component-label', index=MATCH), 'className'),
                          app=app)

        # enable cluster opacity slider when clustering updates
        targeted_callback(lambda _: False,
                          Input(dict(type='clustering-status', pattern=True), 'data'),
                          Output(self._cluster_overlay_opacity.id, 'disabled'),
                          app=app)

    def build_item(self, i):
        checkbox = dbc.Checkbox(id=dict(type='decomposition-component-toggle', index=i),
                                 checked=(i==0),
                                 style=dict(display='none'),
                                 # className='btn-group-vertical col-sm-auto',
                                 # labelClassName="btn btn-secondary",
                                 # labelCheckedClassName="active",
                                )
        check_label = dbc.Label(id=dict(type='decomposition-component-label', index=i),
                                children=f'{i+1}',
                                className='btn btn-secondary' + (' active' if checkbox.checked else ''),
                                html_for=stringify_id(checkbox.id))
        color_scale_selector = ColorScaleSelector({'type': 'decomposition-color-scale-selector',
                            'index': i},
                           values=decomposition_color_scales,
                           value=decomposition_color_scales[
                               i % len(decomposition_color_scales)],
                           )
        slider = dcc.Slider(
            id={'type': 'component-opacity',
                'index': i},
            min=0,
            max=1,
            step=.1,
            value=.5 if i else 1,
            className='centered-slider col',
            disabled=True if i else False
        )
        self._checkboxes.append(checkbox)
        self._check_labels.append(check_label)
        self._component_color_scale_selectors.append(color_scale_selector)
        self._component_opacity_sliders.append(slider)

        return html.Div([checkbox, check_label, color_scale_selector, slider], className='row well', style=dict())

    def disable_slider(self, checked):
        return not checked

    def update_items(self, n_items):
        items = self.decomposition_selector_layout.children

        # set display style for each existing item
        for i, item in enumerate(items):
            item.style['display'] = 'none' if i > n_items else 'flex'

        # if more items are required, build them
        if n_items > len(items):
            for i in range(len(items), n_items):
                items.append(self.build_item(i))

        return items

    def update_check_state(self, checked):
        className = set(next(iter(dash.callback_context.states.values())).split(' '))
        if checked:
            className.add('active')
        else:
            className.remove('active')
        return ' '.join(list(className))


class DecompositionGraph(SliceGraph):
    title = 'Decomposition Maps'

    def __init__(self, data, instance_index, cluster_labels, cluster_label_names, bounds, *args, **kwargs):
        self.configuration_panel = DecompositionGraphPanel(instance_index, data.shape[0], cluster_labels)
        self.bounds = bounds

        self._component_traces = self._build_component_heatmaps(data)

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

    def _build_component_heatmaps(self, data):
        traces = []
        for i in range(data.shape[0]):
            color_scale = decomposition_color_scales[i % len(decomposition_color_scales)]
            color_scale = transparent_color_scales.get(color_scale, color_scale)

            trace = go.Heatmap(z=np.asarray(data[i]),
                               colorscale=color_scale,
                               y0=self.bounds[1][0],
                               dy=(self.bounds[1][1] - self.bounds[1][0]) / (data.shape[1] - 1),
                               x0=self.bounds[2][0],
                               dx=(self.bounds[2][1] - self.bounds[2][0]) / (data.shape[2] - 1),
                               visible=(i == 0),
                               opacity=.5 if i else 1,
                               )

            traces.append(trace)
        return traces

    def rebuild_component_heatmaps(self, data):
        for trace in self._component_traces:
            self._traces.remove(trace)
        self._component_traces = self._build_component_heatmaps(data)
        self._traces = self._component_traces + self._traces
        return self._update_figure()

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
        targeted_callback(self.show_component,
                          Input(dict(type='decomposition-component-toggle', index=ALL), 'checked'),
                          Output(self.id, 'figure'),
                          app=app)

        # Update the color scale when new item is selected
        targeted_callback(self.set_color_scale,
                          Input({'type': 'decomposition-color-scale-selector', 'index': ALL}, 'label'),
                          Output(self.id, 'figure'),
                          app=app)

        # update cluster overlay opacity
        targeted_callback(self.update_opacity,
                          Input(self.configuration_panel._cluster_overlay_opacity.id, 'value'),
                          Output(self.id, 'figure'),
                          app=app)

    def set_color_scale(self, color_scale):
        i = int(re.findall('(?<="index":)\\d+(?=,)', dash.callback_context.triggered[0]['prop_id'])[0])
        color_scale = transparent_color_scales.get(color_scale, color_scale)
        self._component_traces[i].colorscale = color_scale

        return self._update_figure()

    def _opacity_slider(self, i):
        return self.configuration_panel._component_opacity_sliders[i]

    def _color_scale_selector(self, i):
        return self.configuration_panel._component_color_scale_selectors[i]

    def set_component_opacity(self, value):
        i = int(re.findall('(?<="index":)\\d+(?=,)', dash.callback_context.triggered[0]['prop_id'])[0])
        self._opacity_slider(i).value = value
        self._update_opacity()
        return self._update_figure()

    def _update_opacity(self):
        # Get a sum of all enabled slider values minus the first enabled value
        total = 0
        for slider in self.configuration_panel._component_opacity_sliders:
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

    def show_component(self, checked):
        i = int(re.findall('(?<="index":)\\d+(?=,)', dash.callback_context.triggered[0]['prop_id'])[0])
        if i > len(self._component_traces)-1:
            raise PreventUpdate

        trace = self._component_traces[i]
        trace.visible = checked
        trace.showscale = len(list(filter(lambda trace: trace.visible, self._component_traces))) < 2
        self._update_opacity()

        return self._update_figure()
