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
    title = 'Optical Image'

    def __init__(self, map_data, optical_data, bounds, cluster_labels, cluster_label_names, parent, slice_axis=0, traces=None, shapes=None, **kwargs):
        #
        # default_slice_index = 0
        self._map_data = map_data
        #
        # if optical_data.ndim == 2:
        #     optical_data = np.expand_dims(optical_data, 0)
        #
        # # Create traces (i.e. 'glyphs') that will comprise a plotly Figure
        # optical_graph_bounds = dict(y0=bounds[1][0],
        #                             dy=(bounds[1][1]-bounds[1][0])/optical_data.shape[1],
        #                             x0=bounds[2][0],
        #                             dx=(bounds[2][1]-bounds[2][0])/optical_data.shape[2])
        # map_graph_bounds = dict(y0=bounds[1][0],
        #                         dy=(bounds[1][1]-bounds[1][0])/map_data.shape[1],
        #                         x0=bounds[2][0],
        #                         dx=(bounds[2][1]-bounds[2][0])/map_data.shape[2])
        # # Template for custom hover text
        # x_label = kwargs.get('xaxis_title', '')
        # y_label = kwargs.get('yaxis_title', '')
        # i_label = 'I'
        # extra_kwargs = {}
        # if cluster_label_names is not None and cluster_labels is not None:
        #     extra_kwargs['text'] = np.asarray(cluster_label_names)[cluster_labels]
        #     hovertemplate = f'{x_label}: %{{x}}<br />{y_label}: %{{y}}<br />{i_label}: %{{z}}<br />Label: %{{text}}<extra></extra>'
        # else:
        #     hovertemplate = f'{x_label}: %{{x}}<br />{y_label}: %{{y}}<br />{i_label}: %{{z}}<extra></extra>'
        # self._image = go.Heatmap(z=np.asarray(optical_data[default_slice_index]),
        #                          colorscale='viridis',
        #                          hovertemplate=hovertemplate,
        #                          **optical_graph_bounds,
        #                          **extra_kwargs
        #                          )
        # self._selection_mask = go.Heatmap(z=np.ones(map_data[0].shape) * np.NaN,
        #                                   colorscale='reds',
        #                                   opacity=0.3,
        #                                   showscale=False,
        #                                   hoverinfo='skip',
        #                                   **map_graph_bounds
        #                                   )
        # x, y = np.meshgrid(np.linspace(bounds[2][0], bounds[2][1], map_data.shape[2]),
        #                    np.linspace(bounds[1][0], bounds[1][1], map_data.shape[1]))
        #
        # # This dummy scatter trace is added to support lasso selection
        # self._dummy_scatter = go.Scattergl(x=x.ravel(),
        #                                    y=y.ravel(),
        #                                    mode='markers',
        #                                    marker={'color': 'rgba(0,0,0,0)'},
        #                                    hoverinfo='skip'
        #                                    )
        # # Add another transparent heatmap overlay for labels
        # self._clusters = go.Heatmap(z=np.ones(map_data[0].shape) * np.NaN,
        #                             colorscale='Portland',
        #                             **map_graph_bounds,
        #                             opacity=0.3,
        #                             showscale=False,
        #                             hoverinfo='skip',
        #                             )
        # if cluster_labels is not None:
        #     self._clusters.z = cluster_labels  # NaNs are transparent
        #

        # map_data is passed here rather than optical_data as a matter of convenience in class composition
        # All the other traces are scaled according to map_data's shape; the image trace is swapped out next
        super(OpticalGraph, self).__init__(map_data,
                                           bounds,
                                           cluster_labels,
                                           cluster_label_names,
                                           parent,
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
        self._traces.insert(0, self._image)
        self.figure = self._update_figure()

    def _id(self):
        _id = super(OpticalGraph, self)._id()
        _id['subtype'] = 'optical'
        return _id


    # NOTE: THIS method is overridden because the mask shape must be based on the map_data rather than the optical_data
    # TODO: Refactor everything
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