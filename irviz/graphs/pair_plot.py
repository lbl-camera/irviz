import dash
import dash_core_components as dcc
import numpy as np
from dash.dependencies import Output, Input, ALL
from dash.exceptions import PreventUpdate
from plotly import graph_objects as go

from irviz.utils.dash import targeted_callback
__all__ = ['PairPlotGraph']


class PairPlotGraph(dcc.Graph):
    title = 'Pair Plot'

    def __init__(self, data, cluster_labels, cluster_label_names, parent):
        # Track if the selection help has been displayed yet, don't want to annoy users
        self._selection_help_displayed_already = False

        # Cache our data and parent for use in the callbacks
        self._data = data
        self._parent = parent
        self._xaxis_title = self._yaxis_title = ''
        self._cluster_labels = cluster_labels
        self._cluster_label_names = cluster_label_names
        self.traces = []

        figure = self._update_figure()
        super(PairPlotGraph, self).__init__(figure=figure,
                                            id=self._id(),
                                            className='col-lg-3 p-0',
                                            responsive=True,
                                            style=dict(display='flex',
                                                       flexDirection='row',
                                                       height='100%',
                                                       minHeight='450px'),)

    def _id(self):
        return {'type': 'pair_plot',
                'index': self._parent._instance_index}

    def register_callbacks(self):
        # Set up callbacks
        # ----------------

        # When the parent viewer's 'spectra_graph' is clicked
        #     we need to update the internal Figure for this Graph
        # When MapGraph is lasso'd, show that selection here too
        # Note: this can't be a targeted callback, since multiple values are required
        self._parent._app.callback(
            Output(self.id, 'figure'),
            Input(self._parent.decomposition_component_1.id, 'value'),
            Input(self._parent.decomposition_component_2.id, 'value'),
            Input({'type': 'slice_graph',
                   'subtype': ALL,
                   'index': self._parent._instance_index},
                  'selectedData'),
        )(self.show_pair_plot)

        # Set up selection tool callbacks
        targeted_callback(self._show_selection_info,
                          Input(self.id, 'selectedData'),
                          Output(self._parent.info_content.id, 'children'),
                          app=self._parent._app)

        # Set up help notifications for selection tools
        targeted_callback(self._update_selection_help_text,
                          Input(self.id, 'selectedData'),
                          Output(self._parent.notifier.id, 'children'),
                          app=self._parent._app)

        # Wire-up visibility toggle
        targeted_callback(self._set_visibility,
                          Input(self._parent.graph_toggles.id, 'value'),
                          Output(self.id, 'style'),
                          app=self._parent._app)

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

    def show_pair_plot(self, component1, component2, selectedData):
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
            if '"type":"slice_graph"' in triggered[0]['prop_id'] and triggered[0]['value'] is not None:
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

        self._xaxis_title = f'Component #{component1+1}'
        self._yaxis_title = f'Other components' if multi_mode else f'Component #{component2+1}'

        return self._update_figure()

    @staticmethod
    def _indexes_from_selection(selection):
        return list(map(lambda point: point['pointIndex'], selection['points']))

    @staticmethod
    def _set_visibility(switches_value):
        if 'show_pair_plot' in switches_value:
            return {'display': 'block'}
        else:
            return {'display': 'none'}
