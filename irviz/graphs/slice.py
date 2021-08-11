import numpy as np
import dash_core_components as dcc
from dash.dependencies import Input, Output, ALL, State
from dash.exceptions import PreventUpdate
from plotly import graph_objects as go
import dash

from ryujin.utils.dash import targeted_callback
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

    def __init__(self, data, instance_index, cluster_labels=None, cluster_label_names=None, bounds=None, slice_axis=0,
                 traces=None, shapes=None, graph_kwargs=None, **kwargs):
        # Normalize bounds
        if bounds is None or np.asarray(bounds).shape != (
                3, 2):  # bounds should contain a min/max pair for each dimension
            bounds = [[0, data.shape[0] - 1],
                      [0, data.shape[1] - 1],
                      [0, data.shape[2] - 1]]
        bounds = np.asarray(bounds)

        # Cache our data for use in the callbacks
        self._data = data
        self._instance_index = instance_index
        self._bounds = bounds
        self._traces = traces or []
        self._annotation_traces = []
        self._shapes = shapes or []
        self._cluster_labels = cluster_labels
        self.xaxis_title = kwargs.pop('xaxis_title', '')
        self.yaxis_title = kwargs.pop('yaxis_title', '')

        # Build shapes
        self._h_line = go.layout.Shape(type='line',
                                       # width=3,
                                       xref='paper',
                                       yref='y',
                                       x0=0,
                                       x1=1,
                                       y0=(bounds[1][1] + bounds[1][0]) / 2,
                                       y1=(bounds[1][1] + bounds[1][0]) / 2)
        self._v_line = go.layout.Shape(type='line',
                                       xref='x',
                                       yref='paper',
                                       x0=(bounds[2][1] + bounds[2][0]) / 2,
                                       x1=(bounds[2][1] + bounds[2][0]) / 2,
                                       y0=0,
                                       y1=1)

        extra_kwargs = {}
        if cluster_label_names is not None and cluster_labels is not None:
            extra_kwargs['text'] = np.asarray(cluster_label_names)[cluster_labels]
            hovertemplate = f'{self.xaxis_title}: %{{x}}<br />{self.yaxis_title}: %{{y}}<br />I: %{{z}}<br />Label: %{{text}}<extra></extra>'
        else:
            hovertemplate = f'{self.xaxis_title}: %{{x}}<br />{self.yaxis_title}: %{{y}}<br />I: %{{z}}<extra></extra>'

        default_slice_index = (data.shape[0] - 1) // 2
        self._image = self._get_image_trace(data[default_slice_index],
                                            bounds,
                                            hovertemplate=hovertemplate,
                                            **extra_kwargs)

        self._selection_mask = self._get_image_trace(np.ones_like(data[0]) * np.NaN,
                                                     bounds,
                                                     colorscale='reds',
                                                     opacity=0.3,
                                                     showscale=False,
                                                     hoverinfo='skip',
                                                     name='selection')

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
        self._clusters = self._get_image_trace(np.ones_like(data[0]) * np.NaN,
                                               bounds,
                                               colorscale='Portland',
                                               opacity=0.3,
                                               showscale=False,
                                               hoverinfo='skip', )

        if cluster_labels is not None:
            self._clusters.z = cluster_labels  # NaNs are transparent

        self._shapes.extend([self._h_line, self._v_line])
        self._traces = (traces or []) + [self._dummy_scatter, self._image, self._selection_mask, self._clusters]

        figure = self._update_figure()
        super(SliceGraph, self).__init__(figure=figure,
                                         id=self._id(instance_index),
                                         **graph_kwargs or {})

    def _id(self, instance_index):
        return {'type': 'slice_graph',
                'subtype': ...,
                'index': instance_index}

    def _get_image_trace(self, data, bounds, **extra_kwargs):
        graph_bounds = dict(y0=bounds[1][0],
                            dy=(bounds[1][1] - bounds[1][0]) / (data.shape[0] - 1),
                            x0=bounds[2][0],
                            dx=(bounds[2][1] - bounds[2][0]) / (data.shape[1] - 1))

        if data.ndim == 3 and data.shape[-1] == 3:
            graph_object = go.Image
        else:
            graph_object = go.Heatmap
            if 'colorscale' not in extra_kwargs:
                extra_kwargs['colorscale'] = 'viridis'

        return graph_object(z=np.asarray(data),
                            **graph_bounds,
                            **extra_kwargs
                            )

    def init_callbacks(self, app):
        # When any SliceGraph is clicked, update its x,y slicer lines
        targeted_callback(self.show_click,
                          Input({'type': 'slice_graph',
                                 'subtype': ALL,
                                 'index': self._instance_index},
                                'clickData'),
                          Output(self.id, 'figure'),
                          app=app)

        # When points are selected in the pair plot, show them here
        targeted_callback(self._show_selection_mask,
                          Input({'type': 'pair_plot',
                                 'wildcard': ALL,
                                 'index': self._instance_index}, 'selectedData'),
                          Output(self.id, 'figure'),
                          app=app)

        # When any SliceGraph is lasso'd, update the selection mask
        targeted_callback(self._show_selection_mask,
                          Input({'type': 'slice_graph',
                                 'subtype': ALL,
                                 'index': self._instance_index},
                                'selectedData'),
                          Output(self.id, 'figure'),
                          app=app)

        # Share zoom state
        targeted_callback(self.sync_zoom,
                          Input({'type': 'slice_graph',
                                 'subtype': ALL,
                                 'index': self._instance_index},
                                'relayoutData'),
                          Output(self.id, 'figure'),
                          State({'type': 'slice_graph',
                                 'subtype': ALL,
                                 'index': self._instance_index},
                                'figure'),
                          app=app)

        # update with annotations
        targeted_callback(self.update_annotations,
                          Input({'type': 'slice_annotations',
                                 'index': self._instance_index,
                                 'wildcard': ALL},
                                'children'),
                          Output(self.id, 'figure'),
                          app=app)

        # update cluster overlay opacity
        targeted_callback(self.update_opacity,
                          Input(self.configuration_panel._cluster_overlay_opacity.id, 'value'),
                          Output(self.id, 'figure'),
                          app=app)

    def update_opacity(self, value):
        self._clusters.opacity = value

        return self._update_figure()

    def sync_zoom(self, relayoutData):
        figure = self._update_figure()

        if f'"type":"{self.id["type"]}"' in dash.callback_context.triggered[0]['prop_id']:
            # Don't self-update
            if f'"subtype":"{self.id["subtype"]}"' in dash.callback_context.triggered[0]['prop_id']:
                raise PreventUpdate

            try:
                figure['layout']['xaxis']['range'] = [relayoutData['xaxis.range[0]'],
                                                      relayoutData['xaxis.range[1]']]
                figure['layout']['yaxis']['range'] = [relayoutData['yaxis.range[0]'],
                                                      relayoutData['yaxis.range[1]']]
                figure['layout']['xaxis']['autorange'] = False
                figure['layout']['yaxis']['autorange'] = False
            except KeyError:  # ignore when we haven't already zoomed
                pass

            return figure

        raise PreventUpdate

    def set_color_scale(self, color_scale):
        if hasattr(self._image, 'colorscale'):
            self._image.colorscale = color_scale
            return self._update_figure()

        raise PreventUpdate

    def set_clusters_visibility(self, value):
        self._clusters.visible = 'show_clusters' in value

        return self._update_figure()

    def update_slice(self, spectra_graph_click_data):
        slice = spectra_graph_click_data["points"][0]["x"]
        slice_index = nearest_bin(slice, self._bounds[0], self._data.shape[0])
        self._image.z = np.asarray(self._data[slice_index])

        return self._update_figure()

    def _update_figure(self):
        fig = go.Figure(self._traces + self._annotation_traces)
        fig.update_layout(title=self.title,
                          xaxis_title=self.xaxis_title,
                          yaxis_title=self.yaxis_title,
                          )
        fig.update_yaxes(autorange=True,
                         constrain='range')
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

    @staticmethod
    def _set_visibility(checked):
        if checked:
            return {'display': 'block'}
        else:
            return {'display': 'none'}

    def update_annotations(self, *_):
        return self._update_figure()

    def add_annotation(self, annotation):
        colorscale = None
        if 'color' in annotation:
            colorscale = [[0, 'rgb(0,0,0)'],
                          [1, annotation['color']]]
        opacity = annotation.get('opacity', 0.3)
        annotation_trace = self._get_image_trace(annotation['mask'],
                                                 bounds=self._bounds,
                                                 colorscale=colorscale,
                                                 opacity=opacity,
                                                 showscale=False,
                                                 hoverinfo='skip',
                                                 name=annotation['name'],
                                                 meta={'annotation_index': annotation['annotation_index']}
                                                 )
        self._annotation_traces.append(annotation_trace)

        return annotation_trace

    def remove_annotation(self, index):
        remaining_annotations = list(filter(lambda a: a.meta['annotation_index'] != index, self._annotation_traces))
        self._annotation_traces = remaining_annotations
        return self._update_figure()

    @property
    def annotations(self):
        annotations = []
        for trace in self._annotation_traces:
            annotation = {'name': trace.name,
                          'mask': trace.z == 1}
            annotations.append(annotation)
        return annotations

    @property
    def position(self):
        """The current spatial position of the crosshair"""
        return self._v_line.x0, self._h_line.y0

    @property
    def position_index(self):
        """The spatial position of the current spectrum as an index (y, x)"""
        return nearest_bin(self._x_line.y0, self._bounds[1], self._ata.shape[1]), \
               nearest_bin(self._v_line.x0, self._bounds[2], self._data.shape[2])

    @property
    def selection(self):
        """A mask array representing the current spatial selection region"""
        return self._selection_mask.z

    @property
    def selection_indices(self):
        """The indices of all currently selected points, returned as (y, x)"""
        return np.argwhere(self._selection_mask.z)
