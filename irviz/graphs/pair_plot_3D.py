import dash
import dash_bootstrap_components as dbc
import numpy as np
import phonetic_alphabet
from dash import dcc, html
from dash._utils import stringify_id
from dash.dependencies import Output, Input, ALL, State
from dash.exceptions import PreventUpdate
from plotly import graph_objects as go, colors

from irviz.utils.math import nearest_bin
from ryujin.components import Panel
from ryujin.utils.dash import targeted_callback

__all__ = ['PairPlot3DGraph']


class PairPlot3DGraphPanel(Panel):
    def __init__(self, instance_index, component_count):
        radio_kwargs = dict(className='btn-group-vertical col-sm-auto',
                            labelClassName="btn btn-secondary",
                            labelCheckedClassName="active",
                            options=[{'label': f'{i + 1}', 'value': i}
                                     for i in range(component_count)]

                            )

        radio_kwargs['className'] = 'btn-group'  # wipe out other classes

        self._decomposition_component_1 = dbc.RadioItems(id='3D-component-selector-1', value=0, **radio_kwargs)
        self._decomposition_component_2 = dbc.RadioItems(id='3D-component-selector-2', value=1, **radio_kwargs)
        radio_kwargs = radio_kwargs.copy()
        radio_kwargs['options'] = radio_kwargs['options'].copy() + [{'label': 'ALL', 'value': 'ALL'}]
        self._decomposition_component_3 = dbc.RadioItems(id='3D-component-selector-3', value=2, **radio_kwargs)

        pair_plot_component_selector = html.Div(
            [
                self._decomposition_component_1,
                html.Br(),
                self._decomposition_component_2,
                html.Br(),
                self._decomposition_component_3,
            ],
            className='radio-group mb-3',
        )

        self.visibility_toggle = dbc.Checkbox(id=dict(type='3D-pair-plot-visibility', instance_index=instance_index), value=True)

        children = [html.Div([self.visibility_toggle,
                                   dbc.Label('Show 3D Pair Plot')], className='mb-3'),
                    html.Div([dbc.Label('Shown Components'),
                                   pair_plot_component_selector], className='mb-3')]

        super(PairPlot3DGraphPanel, self).__init__('3D Pair Plot', children)

    def init_callbacks(self, app):
        super(PairPlot3DGraphPanel, self).init_callbacks(app)

        # Change number of items when decomposition changes
        targeted_callback(self.update_items_1,
                          Input(dict(type='decomposition-status', pattern=ALL), 'data'),
                          Output(self._decomposition_component_1.id, 'options'),
                          app=app)
        targeted_callback(self.update_items_1,
                          Input(dict(type='decomposition-status', pattern=ALL), 'data'),
                          Output(self._decomposition_component_2.id, 'options'),
                          app=app)
        targeted_callback(self.update_items_2,
                          Input(dict(type='decomposition-status', pattern=ALL), 'data'),
                          Output(self._decomposition_component_3.id, 'options'),
                          app=app)

    def update_items_1(self, component_count):
        options = [{'label': f'{i + 1}', 'value': i}
                   for i in range(component_count)]
        return options

    def update_items_2(self, component_count):
        options = [{'label': f'{i + 1}', 'value': i}
                   for i in range(component_count)] + [{'label': 'ALL', 'value': 'ALL'}]
        return options


class PairPlot3DGraph(dcc.Graph):
    title = '3D Pair Plot'

    def __init__(self, instance_index, data, bounds, cluster_labels, cluster_label_names, graph_kwargs=None, **kwargs):
        self.configuration_panel = PairPlot3DGraphPanel(instance_index, data.shape[0])

        # Track if the selection help has been displayed yet, don't want to annoy users
        self._selection_help_displayed_already = False

        # Cache our data for use in the callbacks
        self._instance_index = instance_index
        self._data = data
        self._xaxis_title = self._yaxis_title = self._zaxis_title = ''
        self._cluster_labels = cluster_labels
        self._cluster_label_names = cluster_label_names
        self._bounds = bounds
        self._component_traces = []
        self.crosshair_trace = None
        self._crosshair_index = None
        self.selected_points = None

        # Initialize persistent traces
        self.crosshair_trace = go.Scatter3d(x=[],
                                            y=[],
                                            z=[],
                                            showlegend=False,
                                            mode='markers',
                                            marker={'color': 'white',
                                                    'size': 10,
                                                    'line': {'width': 2}},
                                            hoverinfo='skip')

        figure = self.show_pair_plot(0, 1, 2)
        super(PairPlot3DGraph, self).__init__(figure=figure,
                                              id=self._id(),
                                              **graph_kwargs or {})

    def _id(self):
        return {'type': 'pair_plot_3d',
                'index': self._instance_index,
                'wildcard': True}  # The wildcard field is only here to enable 0-match patterns

    def init_callbacks(self, app):
        # Set up callbacks
        # ----------------

        # When the parent viewer's 'spectra_graph' is clicked
        #     we need to update the internal Figure for this Graph
        # When MapGraph is lasso'd, show that selection here too
        # when component ids change update plotted data
        targeted_callback(self.show_pair_plot,
                          Input(self.configuration_panel._decomposition_component_1.id, 'value'),
                          Output(self.id, 'figure'),
                          State(self.configuration_panel._decomposition_component_1.id, 'value'),
                          State(self.configuration_panel._decomposition_component_2.id, 'value'),
                          State(self.configuration_panel._decomposition_component_3.id, 'value'),

                          app=app)
        targeted_callback(self.show_pair_plot,
                          Input(self.configuration_panel._decomposition_component_2.id, 'value'),
                          Output(self.id, 'figure'),
                          State(self.configuration_panel._decomposition_component_1.id, 'value'),
                          State(self.configuration_panel._decomposition_component_2.id, 'value'),
                          State(self.configuration_panel._decomposition_component_3.id, 'value'),

                          app=app)
        targeted_callback(self.show_pair_plot,
                          Input(self.configuration_panel._decomposition_component_3.id, 'value'),
                          Output(self.id, 'figure'),
                          State(self.configuration_panel._decomposition_component_1.id, 'value'),
                          State(self.configuration_panel._decomposition_component_2.id, 'value'),
                          State(self.configuration_panel._decomposition_component_3.id, 'value'),

                          app=app)

        # When any SliceGraph is clicked, update its highlighted point
        targeted_callback(self.show_click,
                          Input({'type': 'slice_graph',
                                 'subtype': ALL,
                                 'index': self._instance_index},
                                'clickData'),
                          Output(self.id, 'figure'),
                          State(self.configuration_panel._decomposition_component_1.id, 'value'),
                          State(self.configuration_panel._decomposition_component_2.id, 'value'),
                          State(self.configuration_panel._decomposition_component_3.id, 'value'),
                          app=app)

        # When any any SliceGraph is lasso'd, show that selection here too
        targeted_callback(self.show_selection,
                          Input({'type': 'slice_graph',
                                 'subtype': ALL,
                                 'index': self._instance_index},
                                'selectedData'),
                          Output(self.id, 'figure'),
                          State(self.configuration_panel._decomposition_component_1.id, 'value'),
                          State(self.configuration_panel._decomposition_component_2.id, 'value'),
                          State(self.configuration_panel._decomposition_component_3.id, 'value'),
                          app=app)

        # Set up help notifications for selection tools
        targeted_callback(self._update_selection_help_text,
                          Input(self.id, 'selectedData'),
                          Output({'type': 'notifier',
                                  'index': self._instance_index},
                                 'children'),
                          app=app)

        # Wire-up visibility toggle
        targeted_callback(self._set_visibility,
                          Input(self.configuration_panel.visibility_toggle.id, 'value'),
                          Output(self.id, 'style'),
                          app=app)

    def _update_selection_help_text(self, selected_data):
        if not self._selection_help_displayed_already:
            self._selection_help_displayed_already = True
            return "Double-click with selection tool to unselect all points."
        else:
            raise PreventUpdate

    def _show_selection_info(self, selected_data):
        if not selected_data:
            return "info"
        return str(list(map(lambda point: point['pointIndex'], selected_data['points'])))

    def _update_figure(self):
        """ Remake the figure to force a display update """
        fig = go.Figure(self._component_traces + [self.crosshair_trace])
        fig.update_layout(title=self.title,
                          scene=dict(xaxis_title=self._xaxis_title,
                                     yaxis_title=self._yaxis_title,
                                     zaxis_title=self._zaxis_title), )
        return fig

    def show_selection(self, selected_points):
        self.selected_points = [point['pointIndex'] for point in selected_points['points']] \
            if selected_points is not None else None
        return self.show_pair_plot()

    def show_click(self, click_data, component1=None, component2=None, component3=None):
        if click_data is not None:
            y_index = nearest_bin(click_data["points"][0]["y"], self._bounds[1], self._data.shape[1])
            x_index = nearest_bin(click_data["points"][0]["x"], self._bounds[2], self._data.shape[2])
            self._crosshair_index = np.ravel_multi_index((y_index, x_index), self._data.shape[1:])
        return self.show_pair_plot(component1, component2, component3)

    def show_pair_plot(self, component1=None, component2=None, component3=None):
        if not component3:
            component1, component2, component3, match_components = self._get_components()
        else:
            component1, component2, component3, match_components = self._get_components(component1, component2, component3)

        self._component_traces = []
        multi_mode = component3 == 'ALL'
        cluster_label_mode = len(match_components) == 1 and self._cluster_labels is not None

        if self._crosshair_index is not None:
            x = [self._data[component1].ravel()[self._crosshair_index] for component3 in match_components]
            y = [self._data[component2].ravel()[self._crosshair_index] for component3 in match_components]
            z = [self._data[component3].ravel()[self._crosshair_index] for component3 in match_components]
            self.crosshair_trace.x = x
            self.crosshair_trace.y = y
            self.crosshair_trace.z = z

        self._xaxis_title = f'Component #{component1 + 1}'
        self._yaxis_title = f'Component #{component2 + 1}'
        self._zaxis_title = f'Other components' if len(match_components) > 1 else f'Component #{component3 + 1}'

        for component3 in match_components:
            # Default None - Any non-array value passed to selectedpoints kwarg indicates there is no selection present
            selected_points = self.selected_points
            try:
                x = self._data[component1]
                y = self._data[component2]
                z = self._data[component3]
            except IndexError:
                return self._update_figure()

            self._component_traces.append(go.Scatter3d(x=np.asarray(x.ravel()),
                                                       y=np.asarray(y.ravel()),
                                                       z=np.asarray(z.ravel()),
                                                       mode='markers',
                                                       marker={'color': 'rgba(0,0,0,0)'} if cluster_label_mode else None,
                                                       hoverinfo='skip' if cluster_label_mode else None,
                                                       showlegend=True if multi_mode else False,
                                                       # selectedpoints=selected_points,  # TODO: look into uirevision property to satisfy this
                                                       name=f'Component #{component3 + 1}' if multi_mode else None))

            if cluster_label_mode:
                min_index = 0
                for i, name in enumerate(self._cluster_label_names):
                    label_mask = self._cluster_labels.ravel() == i
                    masked_selected_points = None
                    if selected_points is not None:
                        masked_selected_points = np.asarray(selected_points) - min_index
                        masked_selected_points = masked_selected_points[
                            np.logical_and(0 <= masked_selected_points, masked_selected_points < np.count_nonzero(label_mask))
                        ]
                    trace = go.Scatter3d(x=np.asarray(x.ravel())[label_mask],
                                         y=np.asarray(y.ravel())[label_mask],
                                         z=np.asarray(z.ravel())[label_mask],
                                         name=name,
                                         mode='markers',
                                         marker={'color': colors.qualitative.D3[i % len(colors.qualitative.Vivid)]}
                                         # selectedpoints=masked_selected_points,
                                         )
                    self._component_traces.append(trace)
                    min_index += np.count_nonzero(label_mask)

        return self._update_figure()

    def _get_components(self, component1=None, component2=None, component3=None):
        if component1 is None:
            component1 = dash.callback_context.states[
                stringify_id(self.configuration_panel._decomposition_component_1.id) + '.value']
        if component2 is None:
            component2 = dash.callback_context.states[
                stringify_id(self.configuration_panel._decomposition_component_2.id) + '.value']
        if component3 is None:
            component3 = dash.callback_context.states[
                stringify_id(self.configuration_panel._decomposition_component_3.id) + '.value']
        if component1 is None or component2 is None or component3 is None:
            raise PreventUpdate

        multi_mode = component3 == 'ALL'
        cluster_label_mode = not multi_mode and self._cluster_labels is not None

        match_components = list(range(component1)) + list(range(component1 + 1, self._data.shape[0])) if multi_mode else [
            component3]

        return component1, component2, component3, match_components

    def update_selection(self, click_data):
        component1, component2, component3, match_components = self._get_components()

        y_index = nearest_bin(click_data["points"][0]["y"], self._bounds[1], self._data.shape[1])  # TODO: How does this work in 3D?
        x_index = nearest_bin(click_data["points"][0]["x"], self._bounds[2], self._data.shape[2])
        self._crosshair_index = np.ravel_multi_index((y_index, x_index), self._data.shape[1:])

        if self._crosshair_index is not None:
            x = [self._data[component1].ravel()[self._crosshair_index] for component2 in match_components]
            y = [self._data[component2].ravel()[self._crosshair_index] for component2 in match_components]
            self.crosshair_trace.x = x
            self.crosshair_trace.y = y

        self._xaxis_title = f'Component #{component1 + 1}'
        self._yaxis_title = f'Other components' if len(match_components) > 1 else f'Component #{component2 + 1}'

        return self._update_figure()

    @staticmethod
    def _indexes_from_selection(selection):
        return list(map(lambda point: point['pointIndex'], selection['points']))

    @staticmethod
    def _set_visibility(checked):
        if checked:
            return {'display': 'block'}
        else:
            return {'display': 'none'}

    def set_data(self, data):
        self._data = data

    def set_clustering(self, cluster_labels, label_names=None):
        if label_names is None:
            sel = ~np.isnan(cluster_labels)
            label_names = [phonetic_alphabet.read(chr(65 + i)) for i in range(np.unique(cluster_labels[sel]).size)]
        self._cluster_labels = cluster_labels
        self._cluster_label_names = label_names
