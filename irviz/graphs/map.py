import numpy as np
from dash.dependencies import Input, Output
from plotly import graph_objects as go

from irviz.graphs.slice import SliceGraph
from irviz.utils.dash import targeted_callback
from irviz.utils.math import nearest_bin
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

    def __init__(self, data, bounds, cluster_labels, cluster_label_names, parent, slice_axis=0, traces=None, shapes=None, **kwargs):

        default_slice_index = (data.shape[0] - 1) // 2

        # Create traces (i.e. 'glyphs') that will comprise a plotly Figure
        graph_bounds = dict(y0=bounds[1][0],
                            dy=(bounds[1][1]-bounds[1][0])/data.shape[1],
                            x0=bounds[2][0],
                            dx=(bounds[2][1]-bounds[2][0])/data.shape[2])
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
        self._image = go.Heatmap(z=np.asarray(data[default_slice_index]),
                                 colorscale='viridis',
                                 hovertemplate=hovertemplate,
                                 **graph_bounds,
                                 **extra_kwargs
                                 )
        self._selection_mask = go.Heatmap(z=np.ones(data[0].shape) * np.NaN,
                                          colorscale='reds',
                                          opacity=0.3,
                                          showscale=False,
                                          hoverinfo='skip',
                                          **graph_bounds
                                          )
        x, y = np.meshgrid(np.linspace(bounds[2][0], bounds[2][1], data.shape[2]),
                           np.linspace(bounds[1][0], bounds[1][1], data.shape[1]))

        # This dummy scatter trace is added to support lasso selection
        self._dummy_scatter = go.Scattergl(x=x.ravel(),
                                           y=y.ravel(),
                                           mode='markers',
                                           marker={'color': 'rgba(0,0,0,0)'},
                                           hoverinfo='skip'
                                           )
        # Add another transparent heatmap overlay for labels
        self._clusters = go.Heatmap(z=np.ones(data[0].shape) * np.NaN,
                                    colorscale='Portland',
                                    **graph_bounds,
                                    opacity=0.3,
                                    showscale=False,
                                    hoverinfo='skip',
                                    )
        if cluster_labels is not None:
            self._clusters.z = cluster_labels  # NaNs are transparent

        traces = (traces or []) + [self._dummy_scatter, self._image, self._selection_mask, self._clusters]

        super(MapGraph, self).__init__(data,
                                       bounds,
                                       parent,
                                       slice_axis=slice_axis,
                                       traces=traces,
                                       shapes=shapes,
                                       **kwargs
                                       # config={'modeBarButtonsToAdd': ['lasso2d']}
                                       )

    def register_callbacks(self):
        # When the spectra graph is clicked, update image slicing
        targeted_callback(self.update_slice,
                          Input(self._parent.spectra_graph.id, 'clickData'),
                          Output(self.id, 'figure'),
                          app=self._parent._app)

        # When this SliceGraph itself is clicked, update its x,y slicer lines
        targeted_callback(self.show_click,
                          Input(self.id, 'clickData'),
                          Output(self.id, 'figure'),
                          app=self._parent._app)

        # When the optical graph is clicked show the same position
        targeted_callback(self.show_click,
                          Input(self._parent.optical_graph.id, 'clickData'),
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

    def set_color_scale(self, color_scale):
        self._image.colorscale = color_scale

        return self._update_figure()

    def set_clusters_visibility(self, value):
        self._clusters.visible = 'show_clusters' in value

        return self._update_figure()

    def update_slice(self, spectra_graph_click_data):
        slice = spectra_graph_click_data["points"][0]["x"]
        slice_index = nearest_bin(slice, self._bounds[0], self._data.shape[0])
        self._image.z = np.asarray(self._data[slice_index])

        return self._update_figure()

    def _show_selection_mask(self, selection):
        # Check two cases:
        #     1. selection is None: initial state (no selection) or user has dbl-clicked w/ lasso/selection tool
        #     2. selection['points'] is empty: user has selected no points
        if selection is not None and len(selection['points']) > 0:
            # Get x,y from the raveled indexes
            raveled_indexes = list(map(lambda point: point['pointIndex'],
                                       filter(lambda point: point['curveNumber'] == 0,
                                              selection['points'])))
            mask = np.zeros(self._data[0].shape)
            # Cannot be 0s - must be NaNs (eval to None) so it doesn't affect underlying HeatMap
            mask.fill(np.NaN)
            mask.ravel()[raveled_indexes] = 1
            # Create overlay
            self._selection_mask.z = mask
        else:
            self._selection_mask.z = np.ones(self._data[0].shape) * np.NaN

        return self._update_figure()