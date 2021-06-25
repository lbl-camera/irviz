from itertools import count

import numpy as np
import dash_core_components as dcc
from dash.dependencies import Input, Output, ALL
from plotly import graph_objects as go

from irviz.utils.dash import targeted_callback
from irviz.utils.math import nearest_bin


class SliceGraph(dcc.Graph):
    """Dash Graph for viewing 2D slices of 3D data.

    Parameters
    ----------
    data : dask.array
        3D data array
    parent : html.Div
        The parent object that creates this Graph

    """

    title = ''
    aspect_locked = True

    def __init__(self, data, bounds, cluster_labels, cluster_label_names, parent, slice_axis=0, traces=None, shapes=None, **kwargs):

        # Cache our data and parent for use in the callbacks
        self._data = data
        self._parent = parent
        self._bounds = bounds
        self._traces = traces or []
        self._shapes = shapes or []
        self.xaxis_title = kwargs.pop('xaxis_title', '')
        self.yaxis_title = kwargs.pop('yaxis_title', '')

        # Build shapes
        self._h_line = go.layout.Shape(type='line',
                                       # width=3,
                                       xref='paper',
                                       yref='y',
                                       x0=0,
                                       x1=1,
                                       y0=(bounds[1][1]+bounds[1][0])/2,
                                       y1=(bounds[1][1]+bounds[1][0])/2)
        self._v_line = go.layout.Shape(type='line',
                                       xref='x',
                                       yref='paper',
                                       x0=(bounds[2][1]+bounds[2][0])/2,
                                       x1=(bounds[2][1]+bounds[2][0])/2,
                                       y0=0,
                                       y1=1)

        # Build traces
        default_slice_index = (data.shape[0] - 1) // 2
        graph_bounds = dict(y0=bounds[1][0],
                            dy=(bounds[1][1]-bounds[1][0])/(data.shape[1]-1),
                            x0=bounds[2][0],
                            dx=(bounds[2][1]-bounds[2][0])/(data.shape[2]-1))

        if data.ndim == 4 and data.shape[-1] == 3:
            graph_object = go.Image
        else:
            graph_object = go.Heatmap

        self._image = graph_object(z=np.asarray(data[default_slice_index]),
                                   colorscale='viridis',
                                   # hovertemplate=hovertemplate,
                                   **graph_bounds,
                                   # **extra_kwargs
                                   )

        extra_kwargs = {}
        x_label = kwargs.get('xaxis_title', '')
        y_label = kwargs.get('yaxis_title', '')
        if cluster_label_names is not None and cluster_labels is not None:
            extra_kwargs['text'] = np.asarray(cluster_label_names)[cluster_labels]
            hovertemplate = f'{x_label}: %{{x}}<br />{y_label}: %{{y}}<br />I: %{{z}}<br />Label: %{{text}}<extra></extra>'
        else:
            hovertemplate = f'{x_label}: %{{x}}<br />{y_label}: %{{y}}<br />I: %{{z}}<extra></extra>'

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
                                    hovertemplate=hovertemplate,
                                    hoverinfo='skip',
                                    **extra_kwargs
                                    )
        if cluster_labels is not None:
            self._clusters.z = cluster_labels  # NaNs are transparent

        self._shapes.extend([self._h_line, self._v_line])
        self._traces = (traces or []) + [self._dummy_scatter, self._image, self._selection_mask, self._clusters]

        figure = self._update_figure()
        super(SliceGraph, self).__init__(figure=figure,
                                         id=self._id(),
                                         className='col-lg-3',
                                         responsive=True,
                                         style=dict(display='flex',
                                                    flexDirection='row',
                                                    height='100%',
                                                    minHeight='450px'),
                                         **kwargs)

    def _id(self):
        return {'type': 'slice_graph',
                'subtype': ...,
                'index': self._parent._instance_index}

    def register_callbacks(self):
        # When any SliceGraph is clicked, update its x,y slicer lines
        targeted_callback(self.show_click,
                          Input({'type': 'slice_graph',
                                 'subtype': ALL,
                                 'index': self._parent._instance_index},
                                'clickData'),
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

        # Change color scale from selector
        targeted_callback(self.set_color_scale,
                          Input(self._parent.map_color_scale_selector.id, 'label'),
                          Output(self.id, 'figure'),
                          app=self._parent._app)

        # When points are selected in the pair plot, show them here
        targeted_callback(self._show_selection_mask,
                          Input(self._parent.pair_plot_graph.id, 'selectedData'),
                          Output(self.id, 'figure'),
                          app=self._parent._app)

        # When any SliceGraph is lasso'd, update the selection mask
        targeted_callback(self._show_selection_mask,
                          Input({'type': 'slice_graph',
                                 'subtype': ALL,
                                 'index': self._parent._instance_index},
                                'selectedData'),
                          Output(self.id, 'figure'),
                          app=self._parent._app)

        # Bind the labels toggle to its trace's visibility
        targeted_callback(self.set_clusters_visibility,
                          Input(self._parent.graph_toggles.id, 'value'),
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

    def _update_figure(self):
        fig = go.Figure(self._traces)
        fig.update_layout(title=self.title,
                          xaxis_title=self.xaxis_title,
                          yaxis_title=self.yaxis_title)
        if self.aspect_locked:
            fig.update_yaxes(scaleanchor="x", scaleratio=1)
        for shape in self._shapes:
            fig.add_shape(shape)
        return fig

    def show_click(self, click_data):
        y_index = click_data["points"][0]["y"]
        x_index = click_data["points"][0]["x"]
        self._h_line.y0 = y_index
        self._h_line.y1 = y_index
        self._v_line.x0 = x_index
        self._v_line.x1 = x_index

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

    @property
    def position(self):
        """The current spatial position of the crosshair"""
        return self._v_line.x0, self._h_line.y0