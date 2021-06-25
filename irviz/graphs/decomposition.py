import re

import dash
import numpy as np
from dash.dependencies import Output, Input, ALL
from plotly import graph_objects as go

from irviz.graphs._colors import decomposition_color_scales, transparent_color_scales
from irviz.graphs.slice import SliceGraph
from irviz.utils.dash import targeted_callback
__all__ = ['DecompositionGraph']


class DecompositionGraph(SliceGraph):
    title = 'Decomposition Maps'

    def __init__(self, data, bounds, cluster_labels, cluster_label_names, parent, *args, **kwargs):

        traces = []
        for i in range(data.shape[0]):
            color_scale = decomposition_color_scales[i % len(decomposition_color_scales)]
            color_scale = transparent_color_scales.get(color_scale, color_scale)

            traces.append(go.Heatmap(z=np.asarray(data[i]),
                                 colorscale=color_scale,
                                 y0=bounds[1][0],
                                 dy=(bounds[1][1]-bounds[1][0])/(data.shape[1]-1),
                                 x0=bounds[2][0],
                                 dx=(bounds[2][1]-bounds[2][0])/(data.shape[2]-1),
                                 visible=(i==0),
                                 opacity=.5 if i else 1,
                                 ))

        kwargs['traces'] = traces

        super(DecompositionGraph, self).__init__(data, bounds, cluster_labels, cluster_label_names, parent, *args, **kwargs)

        # Hide the free image trace
        self._image.visible = False

    def _id(self):
        _id = super(DecompositionGraph, self)._id()
        _id['subtype'] = 'decomposition'
        return _id

    def register_callbacks(self):
        super(DecompositionGraph, self).register_callbacks()

        # Wire-up visibility toggle
        self._parent._app.callback(
            Output(self.id, 'style'),
            Input(self._parent.graph_toggles.id, 'value')
        )(self._set_visibility)

        # Wire-up opacity sliders
        targeted_callback(self.set_component_opacity,
                          Input({'type': 'component-opacity', 'index': ALL}, 'value'),
                          Output(self.id, 'figure'),
                          app=self._parent._app)

        # Show components when selected
        targeted_callback(self.show_components,
                          Input(self._parent.decomposition_component_selector.id, 'value'),
                          Output(self.id, 'figure'),
                          app=self._parent._app)

        # Show clicked position when this graph is clicked
        targeted_callback(self.show_click,
                          Input(self.id, 'clickData'),
                          Output(self.id, 'figure'),
                          app=self._parent._app)

        # Show clicked position when map graph is clicked
        targeted_callback(self.show_click,
                          Input(self._parent.map_graph.id, 'clickData'),
                          Output(self.id, 'figure'),
                          app=self._parent._app)

        # Show clicked position when optical graph is clicked
        targeted_callback(self.show_click,
                          Input(self._parent.optical_graph.id, 'clickData'),
                          Output(self.id, 'figure'),
                          app=self._parent._app)

        # Update the color scale when new item is selected
        targeted_callback(self.set_color_scale,
                          Input({'type':'color-scale-selector', 'index': ALL}, 'label'),
                          Output(self.id, 'figure'),
                          app=self._parent._app)

    def set_color_scale(self, color_scale):
        i = int(re.findall('(?<="index":)\\d+(?=,)', dash.callback_context.triggered[0]['prop_id'])[0])
        color_scale = transparent_color_scales.get(color_scale, color_scale)
        self._traces[i].colorscale = color_scale

        return self._update_figure()

    def _opacity_slider(self, i):
        return self._parent.component_opacity_sliders.children[i]

    def _color_scale_selector(self, i):
        return self._parent.component_color_scale_selectors.children[i]

    def set_component_opacity(self, value):
        i = int(re.findall('(?<="index":)\\d+(?=,)', dash.callback_context.triggered[0]['prop_id'])[0])
        self._opacity_slider(i).value = value
        self._update_opacity()
        return self._update_figure()

    def _update_opacity(self):
        # Get a sum of all enabled slider values minus the first enabled value
        total = 0
        for slider in self._parent.component_opacity_sliders.children:
            if not slider.disabled:
                total += slider.value

        # Set each trace's opacity to a value proportional to its weight; always set first visible trace's opacity to 1
        bg_set = False
        for i, trace in enumerate(self._traces):
            if trace.visible:
                if not bg_set:
                    trace.opacity = 1
                    bg_set = True
                    continue

                trace.opacity = self._opacity_slider(i).value / total

    def show_components(self, component_indices):
        for i, trace in enumerate(self._traces):
            trace.visible = (i in component_indices)
            trace.showscale = len(component_indices) < 2
            self._opacity_slider(i).disabled = not (i in component_indices)  # TODO: set this in a separate callback that outputs to the slider

        self._update_opacity()

        return self._update_figure()




    @staticmethod
    def _set_visibility(switches_value):
        if 'show_decomposition' in switches_value:
            return {'display':'block'}
        else:
            return {'display':'none'}