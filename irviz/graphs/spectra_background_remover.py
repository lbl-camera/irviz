import copy
import warnings
from functools import cached_property, partial
from typing import List, Callable
import threading

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
from dash._utils import create_callback_id
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from plotly import graph_objects as go

from irviz.components.datalists import AnchorPointList, ParameterSetList, RegionList, ParameterSetValueList
from irviz.components.kwarg_editor import KwargsEditor

from irviz.graphs import SpectraPlotGraph
from ryujin.utils.dash import targeted_callback, remove_callback


# NOTE: DataTable caches its data on the client, so if multiple clients are open, the app might end up serving clients with different state


def empty_callable():
    pass


class SpectraBackgroundRemover(SpectraPlotGraph):

    def __init__(self, *args, background_func: Callable=empty_callable, **kwargs):
        self.background_func = background_func
        self._last_update_parameters_sem = threading.Semaphore()
        self.last_update_parameters = None
        parameter_sets = kwargs.get('parameter_sets', [])
        self._parameter_sets = []

        self._anchor_points_trace = go.Scattergl(x=[],
                                                 y=[],
                                                 name=f'_anchor_points',
                                                 showlegend=False,
                                                 # hoverinfo='skip',
                                                 mode='markers+lines',
                                                 line=dict(dash='dash', color='gray'),
                                                 marker=dict(size=16))
        self._background_corrected_trace = go.Scattergl(x=[],
                                                 y=[],
                                                 name=f'Background Corrected',
                                                 # hoverinfo='skip',
                                                 mode='lines',
                                                 line=dict(dash='dot', color='red'))

        super(SpectraBackgroundRemover, self).__init__(*args,
                                                       traces=[self._anchor_points_trace,
                                                               self._background_corrected_trace],
                                                       **kwargs)
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
        self.anchor_points_list = AnchorPointList(table_kwargs=dict(id='anchor-point-list'))
        self.values_editor = KwargsEditor('background_parameters', background_func)
        self.parameter_set_list = ParameterSetList(table_kwargs=dict(id=dict(type='parameter-set-selector',
                                                                             index=self._instance_index),
                                                                     data=parameter_sets))
        self._previous_parameter_set_row_index = 0

        self._values_tab = dbc.Tab(label="Values",
                                   tab_id=f'parameter-set-values-tab',
                                   label_style={'padding': '0.5rem 1rem'},
                                   children=[self.values_editor])
        self._points_tab = dbc.Tab(label="Points",
                                   tab_id=f'parameter-set-points-tab',
                                   label_style={'padding': '0.5rem 1rem'},
                                   children=[self.anchor_points_list])
        self._regions_tab = dbc.Tab(label="Regions",
                                    tab_id=f'parameter-set-regions-tab',
                                    label_style={'padding': '0.5rem 1rem', },
                                    children=[self.region_list])
        tabs = [self._values_tab, self._points_tab, self._regions_tab]
        self._parameter_set_explorer_content = html.Div(id=f'parameter-set-tabs-content-{self._instance_index}')
        self._parameter_set_explorer_tabs = dbc.Tabs(id=f'parameter-set-tabs-{self._instance_index}',
                                                     active_tab=tabs[0].tab_id,
                                                     children=tabs)
        self.parameter_set_explorer = html.Div([
            self._parameter_set_explorer_tabs,
            self._parameter_set_explorer_content
        ])

        # Initialize parameter set values to the default values of the background_func
        for parameter_set in parameter_sets:
            if not parameter_set['values']:
                parameter_set['values'] = self.values_editor.values
        self.parameter_set_add = dbc.Button('Add Parameter Set', id='parameter-set-add')

    def init_callbacks(self, app):
        # update parameter set anchor points
        targeted_callback(self._add_anchor_points,
                          Input(self.id, 'clickData'),
                          Output(self.anchor_points_list.data_table.id, 'data'),
                          State(self.anchor_points_list.data_table.id, 'data'),
                          app=app)

        # update parameter set anchor region
        targeted_callback(self._add_anchor_region,
                          Input(self.id, 'clickData'),
                          Output(self.region_list.data_table.id, 'data'),
                          State(self.region_list.data_table.id, 'data'),
                          app=app)

        super(SpectraBackgroundRemover, self).init_callbacks(app)

        # Remove click callback; that functionality is reproduced here
        remove_callback(super(SpectraBackgroundRemover, self).plot_click,
                        Input(self.id, 'clickData'),
                        Output(self.id, 'figure'),
                        app=app)

        # Change selection mode
        targeted_callback(self.set_mode,
                          Input(self.selection_mode.id, 'value'),
                          Output(self.id, 'figure'),  # not sure what to output to here; self is convenient?
                          app=app)

        # Update the parameter set list when a different parameter set is selected
        # targeted_callback(self._update_current_parameter_set_data,
        #                   Input(self.parameter_set_list.data_table.id, 'selected_rows'),
        #                   Output(self.parameter_set_list.data_table.id, 'data'),
        #                   State(self.parameter_set_list.data_table.id, 'data'),
        #                   app=app)

        # Chained from above; update data lists when the parameter set list is updated
        targeted_callback(self._update_values_editor,
                          Input(self.parameter_set_list.data_table.id, 'selected_rows'),
                          Output(self.values_editor.id, 'children'),
                          State(self.parameter_set_list.data_table.id, 'data'),
                          app=app)
        targeted_callback(self._update_regions_list,
                          Input(self.parameter_set_list.data_table.id, 'selected_rows'),
                          Output(self.region_list.data_table.id, 'data'),
                          State(self.parameter_set_list.data_table.id, 'data'),
                          app=app)
        targeted_callback(self._update_anchor_points_list,
                          Input(self.parameter_set_list.data_table.id, 'selected_rows'),
                          Output(self.anchor_points_list.data_table.id, 'data'),
                          State(self.parameter_set_list.data_table.id, 'data'),
                          app=app)

        # Adds a new parameter set
        targeted_callback(self.add_parameter_set,
                          Input(self.parameter_set_add.id, 'n_clicks'),
                          Output(self.parameter_set_list.data_table.id, 'data'),
                          State(self.parameter_set_list.data_table.id, 'data'),
                          app=app)

        # update parameter set stash when values change
        targeted_callback(partial(self._stash_parameter_set_data, key='anchor_points'),
                          Input(self.anchor_points_list.data_table.id, 'data'),
                          Output(self.parameter_set_list.data_table.id, 'data'),
                          State(self.parameter_set_list.data_table.id, 'data'),
                          State(self.parameter_set_list.data_table.id, 'selected_rows'),
                          app=app)
        targeted_callback(partial(self._stash_parameter_set_data, key='anchor_regions'),
                          Input(self.region_list.data_table.id, 'data'),
                          Output(self.parameter_set_list.data_table.id, 'data'),
                          State(self.parameter_set_list.data_table.id, 'data'),
                          State(self.parameter_set_list.data_table.id, 'selected_rows'),
                          app=app)
        targeted_callback(self._stash_values,
                          Input(self.values_editor.id, 'n_submit'),
                          Output(self.parameter_set_list.data_table.id, 'data'),
                          State(self.parameter_set_list.data_table.id, 'data'),
                          State(self.parameter_set_list.data_table.id, 'selected_rows'),
                          app=app)
        targeted_callback(self._stash_mask,
                          Input(dict(type='slice_graph',
                                     subtype='map',
                                     index=self._instance_index),
                                'figure'),
                          Output(self.parameter_set_list.data_table.id, 'data'),
                          State(self.parameter_set_list.data_table.id, 'data'),
                          State(self.parameter_set_list.data_table.id, 'selected_rows'),
                          State(dict(type='slice_graph',
                                     subtype='map',
                                     index=self._instance_index),
                                'figure'),
                          app=app)

        # update the background if needed
        targeted_callback(self.update_figure_on_data_change,
                          Input(self.parameter_set_list.data_table.id, 'data'),
                          Output(self.id, 'figure'),
                          State(self.parameter_set_list.data_table.id, 'selected_rows'),
                          app=app)
        targeted_callback(self.update_figure_on_selection_change,
                          Input(self.parameter_set_list.data_table.id, 'selected_rows'),
                          Output(self.id, 'figure'),
                          State(self.parameter_set_list.data_table.id, 'data'),
                          app=app)

        self.parameter_set_list.init_callbacks(app)
        self.values_editor.init_callbacks(app)

    def set_mode(self, value):
        self.selection_mode.value = value

        return self._update_figure()

    def _add_anchor_points(self, click_data):
        self._update_energy_line(click_data)  # NOTE: the returned figure is ignored here; it gets re-gen'd along the chain

        if not self.selection_mode.value == 2:
            raise PreventUpdate

        anchor_trace_index = self.figure.data.index(self._anchor_points_trace)
        index = click_data['points'][0]['pointNumber']
        data = dash.callback_context.states[f'{self.anchor_points_list.data_table.id}.data'] or []

        if click_data['points'][0]['curveNumber'] == anchor_trace_index:
            # Remove point
            # self._anchor_points_trace.x = np.delete(self._anchor_points_trace.x, index)
            # self._anchor_points_trace.y = np.delete(self._anchor_points_trace.y, index)
            del data[index]
        else:
            # Add point
            x = click_data['points'][0]['x']
            y = self._plot.y[index]

            xs = [record['x'] for record in data]
            index = np.searchsorted(xs, [x])[0]
            data.insert(index, {'name': f'Anchor #{next(self.anchor_points_list.point_counter)}',
                                'x': x,
                                'y': y})
            # self._anchor_points_trace.x = np.insert(np.asarray(self._anchor_points_trace.x), index, x)
            # if len(self._anchor_points_trace.y):
            #     self._anchor_points_trace.y = np.insert(np.asarray(self._anchor_points_trace.y), index, y)
            # else:
            #     self._anchor_points_trace.y = [y]
        return data

    def _add_anchor_region(self, click_data):
        if not self.selection_mode.value == 3:
            raise PreventUpdate

        x = click_data['points'][0]['x']

        data = dash.callback_context.states[f'{self.region_list.data_table.id}.data'] or []

        # look for a '_region_start' shape already in the figure indicated the previously clicked position
        last_region = data[-1] if len(data) else {}

        if 'region_max' in last_region and last_region['region_max'] is None:
            region = sorted([last_region['region_min'], x])

            data[-1]['region_min'] = region[0]
            data[-1]['region_max'] = region[1]
        else:
            data += [{'name': f'Region #{next(self.region_list.region_counter)}', 'region_min': x, 'region_max': None}]
        return data

    def _update_region_list(self, clickData):
        return self.region_list.data_table.data

    @cached_property
    def configuration_panel(self):
        children = [
            dbc.Form(dbc.FormGroup([self.selection_mode])),
            dbc.Label("Parameter Sets"),
            self.parameter_set_list,
            self.parameter_set_add,
            html.P(''),
            dbc.Form(dbc.FormGroup([self.parameter_set_explorer])),
        ]
        return 'Background Isolator', children

    def _update_regions(self, data):
        # do nothing special if initing
        if hasattr(self, 'figure'):

            # clear shapes except _region_start
            self.figure.layout.shapes = []

            # repopulate from regionlist
            for region_record in data:
                region_min, region_max = region_record.get('region_min'), region_record.get('region_max')
                if region_min is not None and region_max is not None:
                    self.figure.add_vrect(region_min, region_max, line_width=0, opacity=.3, fillcolor='gray')
                elif region_max is None:
                    self.figure.add_vline(region_min, name='_region_start', line_color="gray")

        return self._update_figure()

    def _update_points(self, data):
        # do nothing special if initing
        if hasattr(self, 'figure'):
            self._anchor_points_trace.x = [record['x'] for record in data]
            self._anchor_points_trace.y = [record['y'] for record in data]
        return self._update_figure()

    def _update_current_parameter_set_data(self, selected_rows):
        # Stash the previous data lists into index at previous location (where record 'selected' is True)
        _id = create_callback_id(State(self.parameter_set_list.data_table.id, 'data'))
        parameter_set_list = dash.callback_context.states[_id] or []

        # Update 'selected' property for record at the newly selected row
        row = selected_rows[0]
        parameter_set_list[row]['selected'] = True

        return parameter_set_list

    def _find_selected_parameter_set(self, parameter_set_list_data=None, row=None) -> (int, dict):
        if parameter_set_list_data is None:
            _id = create_callback_id(State(self.parameter_set_list.data_table.id, 'data'))
            parameter_set_list_data = dash.callback_context.states[_id] or []
        if row is None:
            _id = create_callback_id(State(self.parameter_set_list.data_table.id, 'selected_rows'))
            row = dash.callback_context.states[_id][0]
        return row, parameter_set_list_data[row]

    def _update_regions_list(self, selected_rows):
        return self._find_selected_parameter_set(row=selected_rows[0])[-1]['anchor_regions']

    def _update_anchor_points_list(self, selected_rows):
        return self._find_selected_parameter_set(row=selected_rows[0])[-1]['anchor_points']

    def _update_values_editor(self, selected_rows):
        values = self._find_selected_parameter_set(row=selected_rows[0])[-1]['values']
        # rebuild the child items
        return self.values_editor.build_children(values)

    def add_parameter_set(self, n_clicks):
        _id = create_callback_id(State(self.parameter_set_list.data_table.id, 'data'))
        parameter_set_list = dash.callback_context.states[_id] or []

        current_parameter_set = self._find_selected_parameter_set()[-1]

        new_record = self.parameter_set_list.new_record()
        new_record['values'] = current_parameter_set['values'].copy()

        return parameter_set_list + [new_record]

    def _stash_parameter_set_data(self, data, key):
        _id = create_callback_id(State(self.parameter_set_list.data_table.id, 'data'))
        parameter_set_list = dash.callback_context.states[_id] or []
        current_index = self._find_selected_parameter_set(parameter_set_list)[0]
        parameter_set_list[current_index][key] = data
        return parameter_set_list

    def _stash_values(self, n_submits):
        return self._stash_parameter_set_data(self.values_editor.values, 'values')

    def _stash_mask(self, figure_state):
        selection_mask = next(iter(filter(lambda trace: trace.get('name') == 'selection', figure_state['data'])))['z']

        return self._stash_parameter_set_data(selection_mask, 'map_mask')

    def update_figure_on_data_change(self, data):
        # Also, stash the new data for property access
        self._parameter_sets = data
        parameter_set = self._find_selected_parameter_set(parameter_set_list_data=data)[-1]
        return self._update_background(parameter_set)

    def update_figure_on_selection_change(self, selected_rows):
        parameter_set = self._find_selected_parameter_set(row=selected_rows[0])[-1]
        return self._update_background(parameter_set)

    def _update_background(self, parameter_set):
        try:
            background = self.compute_background(parameter_set)
        except Exception as ex:
            if ex is PreventUpdate:
                raise ex
            print(str(ex))
            # if there was data from earlier, clear it
            if len(self._background_corrected_trace.x):
                self._background_corrected_trace.x = []
                self._background_corrected_trace.y = []
        else:
            self._background_corrected_trace.x = self._plot.x
            self._background_corrected_trace.y = self._plot.y - background

        self._update_points(parameter_set['anchor_points'])  # one extra _update_figure happens here; it is ignored
        return self._update_regions(parameter_set['anchor_regions'])

    def _update_background_on_interval(self, n_intervals):
        parameter_set = self._find_selected_parameter_set()[-1]

        with self._last_update_parameters_sem:
            if parameter_set == self.last_update_parameters:
                raise PreventUpdate

            # stash a copy in case background_func tries to mutate things
            parameter_set_stash = copy.deepcopy(parameter_set)

            figure = self._update_background(parameter_set)
            self.last_update_parameters = parameter_set_stash

        return figure

    def compute_background(self, parameter_set):
        return self.background_func(self._plot.x,
                                self._plot.y,
                                [anchor['x'] for anchor in parameter_set['anchor_points']],
                                parameter_set['anchor_regions'],  # TODO: What should get passed in here?
                                parameter_set['map_mask'],
                                **parameter_set['values'])
