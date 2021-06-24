import numpy as np
from dash.dependencies import Input, Output
from plotly import graph_objects as go

from irviz.graphs.slice import SliceGraph
from irviz.utils.dash import targeted_callback
__all__ = ['OpticalGraph']


class OpticalGraph(SliceGraph):
    """Dash Graph for viewing 2D slices of 3D data.

    Parameters
    ----------
    data : dask.array
        3D data array
    parent : html.Div
        The parent object that creates this Graph

    """
    title = 'IR Spectral Map'

    def __init__(self, map_data, optical_data, bounds, cluster_labels, cluster_label_names, parent, slice_axis=0, traces=None, shapes=None, **kwargs):

        default_slice_index = 0
        self._map_data = map_data

        if optical_data.ndim == 2:
            optical_data = np.expand_dims(optical_data, 0)

        # Create traces (i.e. 'glyphs') that will comprise a plotly Figure
        optical_graph_bounds = dict(y0=bounds[1][0],
                                    dy=(bounds[1][1]-bounds[1][0])/optical_data.shape[1],
                                    x0=bounds[2][0],
                                    dx=(bounds[2][1]-bounds[2][0])/optical_data.shape[2])
        map_graph_bounds = dict(y0=bounds[1][0],
                                dy=(bounds[1][1]-bounds[1][0])/map_data.shape[1],
                                x0=bounds[2][0],
                                dx=(bounds[2][1]-bounds[2][0])/map_data.shape[2])
        # Template for custom hover text
        x_label = kwargs.get('xaxis_title', '')
        y_label = kwargs.get('yaxis_title', '')
        i_label = 'I'
        extra_kwargs = {}
        if cluster_label_names is not None and cluster_labels is not None:
            extra_kwargs['text'] = np.asarray(cluster_label_names)[cluster_labels]
            hovertemplate = f'{x_label}: %{{x}}<br />{y_label}: %{{y}}<br />{i_label}: %{{z}}<br />Label: %{{text}}<extra></extra>'
        else:
            hovertemplate = f'{x_label}: %{{x}}<br />{y_label}: %{{y}}<br />{i_label}: %{{z}}<extra></extra>'
        self._image = go.Heatmap(z=np.asarray(optical_data[default_slice_index]),
                                 colorscale='viridis',
                                 hovertemplate=hovertemplate,
                                 **optical_graph_bounds,
                                 **extra_kwargs
                                 )
        self._selection_mask = go.Heatmap(z=np.ones(map_data[0].shape) * np.NaN,
                                          colorscale='reds',
                                          opacity=0.3,
                                          showscale=False,
                                          hoverinfo='skip',
                                          **map_graph_bounds
                                          )
        x, y = np.meshgrid(np.linspace(bounds[2][0], bounds[2][1], map_data.shape[2]),
                           np.linspace(bounds[1][0], bounds[1][1], map_data.shape[1]))

        # This dummy scatter trace is added to support lasso selection
        self._dummy_scatter = go.Scattergl(x=x.ravel(),
                                           y=y.ravel(),
                                           mode='markers',
                                           marker={'color': 'rgba(0,0,0,0)'},
                                           hoverinfo='skip'
                                           )
        # Add another transparent heatmap overlay for labels
        self._clusters = go.Heatmap(z=np.ones(map_data[0].shape) * np.NaN,
                                    colorscale='Portland',
                                    **map_graph_bounds,
                                    opacity=0.3,
                                    showscale=False,
                                    hoverinfo='skip',
                                    )
        if cluster_labels is not None:
            self._clusters.z = cluster_labels  # NaNs are transparent

        traces = (traces or []) + [self._dummy_scatter, self._image, self._selection_mask, self._clusters]

        super(OpticalGraph, self).__init__(optical_data,
                                           bounds,
                                           parent,
                                           slice_axis=slice_axis,
                                           traces=traces,
                                           shapes=shapes,
                                           **kwargs
                                           # config={'modeBarButtonsToAdd': ['lasso2d']}
                                           )

    def set_color_scale(self, color_scale):
        self._image.colorscale = color_scale

        return self._update_figure()

    def set_clusters_visibility(self, value):
        self._clusters.visible = 'show_clusters' in value

        return self._update_figure()

    def register_callbacks(self):
        # When this SliceGraph itself is clicked, update its x,y slicer lines
        targeted_callback(self.show_click,
                          Input(self.id, 'clickData'),
                          Output(self.id, 'figure'),
                          app=self._parent._app)

        # When the map graph is clicked show the same position
        targeted_callback(self.show_click,
                          Input(self._parent.map_graph.id, 'clickData'),
                          Output(self.id, 'figure'),
                          app=self._parent._app)

        # When the decomposition graph is clicked show the same position
        targeted_callback(self.show_click,
                          Input(self._parent.decomposition_graph.id, 'clickData'),
                          Output(self.id, 'figure'),
                          app=self._parent._app)

        # When points are selected in the pair plot, show them here
        targeted_callback(self._show_selection_mask,
                          Input(self._parent.pair_plot_graph.id, 'selectedData'),
                          Output(self.id, 'figure'),
                          app=self._parent._app)

        # When this SliceGraph is lasso'd, update the selection mask
        targeted_callback(self._show_selection_mask,
                          Input(self.id, 'selectedData'),
                          Output(self.id, 'figure'),
                          app=self._parent._app)

        # Bind the labels toggle to its trace's visibility
        targeted_callback(self.set_clusters_visibility,
                          Input(self._parent.graph_toggles.id, 'value'),
                          Output(self.id, 'figure'),
                          app=self._parent._app)

        # Change color scale from selector
        targeted_callback(self.set_color_scale,
                          Input(self._parent.map_color_scale_selector.id, 'label'),
                          Output(self.id, 'figure'),
                          app=self._parent._app)

    # NOTE: THIS method is overridden because the mask shape must be based on the map_data rather than the optical_data
    # TODO: Refactor everything
    def _show_selection_mask(self, selection):
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

        return self._update_figure()