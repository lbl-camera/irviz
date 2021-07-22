import dash
import numpy as np
from dash.dependencies import Output, Input, ALL
from dash.exceptions import PreventUpdate
from plotly import graph_objects as go
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html

from ryujin.components import Panel
from ryujin.utils.dash import targeted_callback
from irviz.utils.math import nearest_bin
__all__ = ['PairPlotGraph']


class PairPlotGraphPanel(Panel):
    def __init__(self, instance_index, component_count):
        radio_kwargs = dict(className='btn-group-vertical col-sm-auto',
                            labelClassName="btn btn-secondary",
                            labelCheckedClassName="active",
                            options=[{'label': f'{i+1}', 'value': i}
                                     for i in range(component_count)]

                            )

        radio_kwargs['className'] = 'btn-group'  # wipe out other classes

        self._decomposition_component_1 = dbc.RadioItems(id='component-selector-1', value=0, **radio_kwargs)
        radio_kwargs = radio_kwargs.copy()
        radio_kwargs['options'] = radio_kwargs['options'].copy() + [{'label': 'ALL', 'value': 'ALL'}]
        self._decomposition_component_2 = dbc.RadioItems(id='component-selector-2', value=1, **radio_kwargs)

        pair_plot_component_selector = dbc.FormGroup(
            [
                self._decomposition_component_1,
                html.Br(),
                self._decomposition_component_2,
            ],
            className='radio-group',
        )

        self.visibility_toggle = dbc.Checkbox(id=dict(type='pair-plot-visibility', instance_index=instance_index), checked=True)

        children = [dbc.FormGroup([self.visibility_toggle,
                                   dbc.Label('Show Decomposition Image')]),
                    dbc.FormGroup([dbc.Label('Shown Components'),
                                   pair_plot_component_selector])]

        super(PairPlotGraphPanel, self).__init__('Pair Plot', children)

    def init_callbacks(self, app):
        super(PairPlotGraphPanel, self).init_callbacks(app)


class PairPlotGraph(dcc.Graph):
    title = 'Pair Plot'

    def __init__(self, instance_index, data, bounds, cluster_labels, cluster_label_names, graph_kwargs=None):
        self.configuration_panel = PairPlotGraphPanel(instance_index, data.shape[0])

        # Track if the selection help has been displayed yet, don't want to annoy users
        self._selection_help_displayed_already = False

        # Cache our data for use in the callbacks
        self._instance_index = instance_index
        self._data = data
        self._xaxis_title = self._yaxis_title = ''
        self._cluster_labels = cluster_labels
        self._cluster_label_names = cluster_label_names
        self._bounds = bounds
        self.traces = []
        self.crosshair_trace = None
        self._crosshair_index = None

        # Initialize persistent traces
        self.crosshair_trace = go.Scattergl(x=[],
                                            y=[],
                                            showlegend=False,
                                            mode='markers',
                                            marker={'color': 'white',
                                                    'size': 10,
                                                    'line': {'width': 2}},
                                            hoverinfo='skip')

        figure = self._update_figure()
        super(PairPlotGraph, self).__init__(figure=figure,
                                            id=self._id(),
                                            **graph_kwargs or {})

    def _id(self):
        return {'type': 'pair_plot',
                'index': self._instance_index,
                'wildcard': True} # The wildcard field is only here to enable 0-match patterns

    def init_callbacks(self, app):
        # Set up callbacks
        # ----------------

        # When the parent viewer's 'spectra_graph' is clicked
        #     we need to update the internal Figure for this Graph
        # When MapGraph is lasso'd, show that selection here too
        # Note: this can't be a targeted callback, since multiple values are required
        app.callback(
            Output(self.id, 'figure'),
            Input(self.configuration_panel._decomposition_component_1.id, 'value'),
            Input(self.configuration_panel._decomposition_component_2.id, 'value'),
            Input({'type': 'slice_graph',
                   'subtype': ALL,
                   'index': self._instance_index},
                  'selectedData'),
            # When any SliceGraph is clicked, update its x,y slicer lines
            Input({'type': 'slice_graph',
                   'subtype': ALL,
                   'index': self._instance_index},
                  'clickData'),
        )(self.show_pair_plot)

        # Set up help notifications for selection tools
        targeted_callback(self._update_selection_help_text,
                          Input(self.id, 'selectedData'),
                          Output({'type': 'notifier',
                                  'index': self._instance_index},
                                 'children'),
                          app=app)

        # Wire-up visibility toggle
        targeted_callback(self._set_visibility,
                          Input(self.configuration_panel.visibility_toggle.id, 'checked'),
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
        fig = go.Figure(self.traces)
        fig.update_layout(title=self.title,
                          xaxis_title=self._xaxis_title,
                          yaxis_title=self._yaxis_title)
        return fig

    def show_pair_plot(self, component1, component2, selectedData, click_data):
        if component1 is None or component2 is None:
            raise PreventUpdate

        self.traces = []

        multi_mode = component2 == 'ALL'
        cluster_label_mode = not multi_mode and self._cluster_labels is not None

        match_components = list(range(component1)) + list(range(component1+1, self._data.shape[0])) if multi_mode else [component2]

        for component2 in match_components:
            # Default None - Any non-array value passed to selectedpoints kwarg indicates there is no selection present
            selected_points = None
            triggered = dash.callback_context.triggered
            if '"type":"slice_graph"' in triggered[0]['prop_id'] and \
                'selectedData' in triggered[0]['prop_id'] and \
                triggered[0]['value'] is not None:
                # selected data being None indicates that the user has selected data
                # selected data 'points' being empty indicates the user has selected data outside of the region
                selected_points = self._indexes_from_selection(triggered[0]['value'])

            x = self._data[component1]
            y = self._data[component2]

            self.traces.append(go.Scattergl(x=np.asarray(x.ravel()),
                                            y=np.asarray(y.ravel()),
                                            mode='markers',
                                            marker={'color': 'rgba(0,0,0,0)'} if cluster_label_mode else None,
                                            hoverinfo='skip' if cluster_label_mode else None,
                                            showlegend=True if multi_mode else False,
                                            selectedpoints=selected_points,
                                            name=f'Component #{component2+1}' if multi_mode else None))

            if cluster_label_mode:
                min_index = 0
                for i, name in enumerate(self._cluster_label_names):
                    label_mask = self._cluster_labels.ravel() == i
                    masked_selected_points = None
                    if selected_points is not None:
                        masked_selected_points = np.asarray(selected_points) - min_index
                        masked_selected_points = masked_selected_points[np.logical_and(0<=masked_selected_points,
                                                                                       masked_selected_points<np.count_nonzero(label_mask))]
                    trace = go.Scattergl(x=np.asarray(x.ravel())[label_mask],
                                         y=np.asarray(y.ravel())[label_mask],
                                         name=name,
                                         mode='markers',
                                         selectedpoints=masked_selected_points)
                    self.traces.append(trace)
                    min_index += np.count_nonzero(label_mask)

        if 'clickData' in triggered[0]['prop_id']:
            click_data = triggered[0]['value']
            y_index = nearest_bin(click_data["points"][0]["y"], self._bounds[1], self._data.shape[1])
            x_index = nearest_bin(click_data["points"][0]["x"], self._bounds[2], self._data.shape[2])
            self._crosshair_index = np.ravel_multi_index((y_index, x_index), self._data.shape[1:])

        if self._crosshair_index is not None:
            x = [self._data[component1].ravel()[self._crosshair_index] for component2 in match_components]
            y = [self._data[component2].ravel()[self._crosshair_index] for component2 in match_components]
            self.crosshair_trace.x = x
            self.crosshair_trace.y = y

        self.traces.append(self.crosshair_trace)

        self._xaxis_title = f'Component #{component1+1}'
        self._yaxis_title = f'Other components' if multi_mode else f'Component #{component2+1}'

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
