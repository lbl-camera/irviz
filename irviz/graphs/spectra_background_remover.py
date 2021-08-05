import warnings
from functools import cached_property

import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from plotly import graph_objects as go
import numpy as np
import dash

from irviz.components.datalists import RegionList
from irviz.graphs import SpectraPlotGraph
from ryujin.utils.dash import targeted_callback


class SpectraBackgroundRemover(SpectraPlotGraph):

    def __init__(self, *args, **kwargs):
        self._anchor_points_trace = go.Scattergl(x=[],
                                                 y=[],
                                                 name=f'_anchor_points',
                                                 showlegend=False,
                                                 # hoverinfo='skip',
                                                 mode='markers+lines',
                                                 line=dict(dash='dash', color='gray'),
                                                 marker=dict(size=16))
        # kwargs.get('graph_kwargs', {})['config'] = {'edits': {'shapePosition': True},
        #                                             }
        super(SpectraBackgroundRemover, self).__init__(*args, traces=[self._anchor_points_trace], **kwargs)
        self.selection_mode = dbc.RadioItems(id=f'background-selection-mode-{self._instance_index}',
                                             className='btn-group radio-group',
                                             labelClassName='btn btn-secondary',
                                             labelCheckedClassName='active',
                                             options=[{'label': 'Standard Mode', 'value': 1},
                                                      {'label': 'Fixed Point Mode', 'value': 2},
                                                      {'label': 'Fixed Region Mode', 'value': 3}],
                                             value=1
                                             )

        self.region_list = RegionList(table_kwargs=dict(id='region-list'), )

    def init_callbacks(self, app):
        targeted_callback(self._add_anchor_region,
                          Input(self.id, 'clickData'),
                          Output(self.region_list.data_table.id, 'data'),
                          State(self.region_list.data_table.id, 'data'),
                          app=app)

        # # when regionlist is changed, update figure
        targeted_callback(self._update_regions,
                          Input(self.region_list.data_table.id, 'data'),
                          Output(self.id, 'figure'),
                          app=app)

        super(SpectraBackgroundRemover, self).init_callbacks(app)

        # Re-declare click callback, adding state; ignore warning
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            targeted_callback(self.plot_click,
                              Input(self.id, 'clickData'),
                              Output(self.id, 'figure'),
                              State(self.region_list.data_table.id, 'data'),
                              app=app)

        # Change selection mode
        targeted_callback(self.set_mode,
                          Input(self.selection_mode.id, 'value'),
                          Output(self.id, 'figure'),  # not sure what to output to here; self is convenient?
                          app=app)

    def set_mode(self, value):
        self.selection_mode.value = value

        return self._update_figure()

    def plot_click(self, click_data):
        data = dash.callback_context.states[f'{self.region_list.data_table.id}.data'] or []

        if self.selection_mode.value == 1:
            return self._update_energy_line(click_data)
        elif self.selection_mode.value == 2:
            return self._add_anchor_points(click_data)
        elif self.selection_mode.value == 3:
            return self._update_regions(data)
        raise PreventUpdate

    def _add_anchor_points(self, click_data):
        anchor_trace_index = self.figure.data.index(self._anchor_points_trace)
        index = click_data['points'][0]['pointNumber']

        if click_data['points'][0]['curveNumber'] == anchor_trace_index:
            # Remove point
            self._anchor_points_trace.x = np.delete(self._anchor_points_trace.x, index)
            self._anchor_points_trace.y = np.delete(self._anchor_points_trace.y, index)
        else:
            # Add point
            x = click_data['points'][0]['x']
            y = self._plot.y[index]

            index = np.searchsorted(self._anchor_points_trace.x, [x])[0]
            self._anchor_points_trace.x = np.insert(np.asarray(self._anchor_points_trace.x), index, x)
            if len(self._anchor_points_trace.y):
                self._anchor_points_trace.y = np.insert(np.asarray(self._anchor_points_trace.y), index, y)
            else:
                self._anchor_points_trace.y = [y]

        return self._update_figure()

    def _add_anchor_region_shape(self, click_data):
        if not self.selection_mode.value == 2:
            raise PreventUpdate

        x = click_data['points'][0]['x']

        # look for a '_region_start' shape already in the figure indicated the previously clicked position
        region_started = next(filter(lambda shape: shape.name == '_region_start', self.figure.layout.shapes), None)

        if region_started:
            region = sorted([region_started.x0, x])
            shapes = list(self.figure.layout.shapes)
            shapes.remove(region_started)
            self.figure.layout.shapes = shapes
            self.region_list.data_table.data += [{'name': 'New Region #', 'region': region}]
        else:
            self.figure.add_vline(x, name='_region_start')

        return self._update_regions()

    def _add_anchor_region(self, click_data):
        if not self.selection_mode.value == 3:
            raise PreventUpdate

        x = click_data['points'][0]['x']

        data = dash.callback_context.states[f'{self.region_list.data_table.id}.data'] or []

        # look for a '_region_start' shape already in the figure indicated the previously clicked position
        region_started = next(filter(lambda shape: shape.name == '_region_start', self.figure.layout.shapes), None)

        if region_started:
            region = sorted([region_started.x0, x])
            shapes = list(self.figure.layout.shapes)
            shapes.remove(region_started)
            self.figure.layout.shapes = shapes

            data[-1]['region_min'] = region[0]
            data[-1]['region_max'] = region[1]
        else:
            self.figure.add_vline(x, name='_region_start', line_color="gray")
            data += [{'name': f'Region #{next(self.region_list.region_counter)}', 'region_min': x, 'region_max': None}]
        return data

    def _update_region_list(self, clickData):
        return self.region_list.data_table.data

    @cached_property
    def configuration_panel(self):
        return 'Background Isolator', [dbc.Form(dbc.FormGroup([self.selection_mode])), self.region_list]

    def _update_regions(self, data):
        # do nothing special if initing
        if hasattr(self, 'figure'):

            # clear shapes except _region_start
            self.figure.layout.shapes = list(filter(lambda shape: shape.name == '_region_start', self.figure.layout.shapes))

            # repopulate from regionlist
            for region_record in data:
                region_min, region_max = region_record.get('region_min'), region_record.get('region_max')
                if region_min is not None and region_max is not None:
                    self.figure.add_vrect(region_min, region_max, line_width=0, opacity=.3, fillcolor='gray')

        return self._update_figure()
