import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from plotly import graph_objects as go
import numpy as np

from irviz.graphs import SpectraPlotGraph
from ryujin.utils.dash import targeted_callback


class SpectraBackgroundRemover(SpectraPlotGraph):

    def __init__(self, *args, **kwargs):
        self._fixed_points_trace = go.Scattergl(x=[],
                                                y=[],
                                                name=f'_fixed_points',
                                                showlegend=False,
                                                # hoverinfo='skip',
                                                mode='markers+lines',
                                                line=dict(dash='dash', color='gray'),
                                                marker=dict(size=16))
        super(SpectraBackgroundRemover, self).__init__(*args, traces=[self._fixed_points_trace], **kwargs)

        self.mode = 1  # standard mode

        self.selection_mode = dbc.RadioItems(id=f'background-selection-mode-{self._instance_index}',
                                             className='btn-group radio-group',
                                             labelClassName='btn btn-secondary',
                                             labelCheckedClassName='active',
                                             options=[{'label': 'Standard Mode', 'value': 1},
                                                      {'label': 'Point Selection Mode', 'value': 2}],
                                             value=1
                                             )



    def init_callbacks(self, app):
        super(SpectraBackgroundRemover, self).init_callbacks(app)

        # Change selection mode
        targeted_callback(self.set_mode,
                          Input(self.selection_mode.id, 'value'),
                          Output(self.id, 'figure'),  # not sure what to output to here; self is convenient?
                          app=app)

    def set_mode(self, value):
        self.mode = value

        return self._update_figure()

    def plot_click(self, click_data):
        if self.mode == 1:
            return self._update_energy_line(click_data)
        elif self.mode == 2:
            return self._add_fixed_point(click_data)

    def _add_fixed_point(self, click_data):
        fixed_trace_index = self.figure.data.index(self._fixed_points_trace)
        index = click_data['points'][0]['pointNumber']

        if click_data['points'][0]['curveNumber'] == fixed_trace_index:
            # Remove point
            self._fixed_points_trace.x = np.delete(self._fixed_points_trace.x, index)
            self._fixed_points_trace.y = np.delete(self._fixed_points_trace.y, index)
        else:
            # Add point
            x = click_data['points'][0]['x']
            y = self._plot.y[index]

            index = np.searchsorted(self._fixed_points_trace.x, [x])[0]
            self._fixed_points_trace.x = np.insert(np.asarray(self._fixed_points_trace.x), index, x)
            if len(self._fixed_points_trace.y):
                self._fixed_points_trace.y = np.insert(np.asarray(self._fixed_points_trace.y), index, y)
            else:
                self._fixed_points_trace.y = [y]

        return self._update_figure()

    j
        return 'Background Isolator', dbc.Form(dbc.FormGroup([self.selection_mode]))

    # def my_background(self, fixed_points, mask, data, ):