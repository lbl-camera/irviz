from dataclasses import dataclass
from typing import List

import dash_bootstrap_components as dbc
import dash_html_components as html
import numpy as np
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from plotly import graph_objects as go

from irviz.components.datalists import ParameterSetList
from irviz.graphs import SpectraPlotGraph
from ryujin.components.datalist import DataList
from ryujin.utils.dash import targeted_callback


@dataclass
class ParameterSet:
    name: str
    values: dict = dict(),
    map_mask: np.ndarray = None,
    anchor_points: list = None
    regions: list = None


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

        self.mode = 1  # standard mode

        self.selection_mode = dbc.RadioItems(id=f'background-selection-mode-{self._instance_index}',
                                             className='btn-group radio-group',
                                             labelClassName='btn btn-secondary',
                                             labelCheckedClassName='active',
                                             options=[{'label': 'Standard Mode', 'value': 1},
                                                      {'label': 'Selection Mode', 'value': 2},
                                                      {'label': 'Region Selection Mode', 'value': 3}],
                                             value=1
                                             )

        self.region_list = DataList(table_kwargs=dict(id='list',
                                                      columns=[{'name': 'name', 'id': 'name'},
                                                             {'name': 'row', 'id': 'row'}],
                                                      data=[{'name':'Test', 'row':1},
                                                          {'name':'Test', 'row':2}],
                                                      row_deletable=True,
                                                      ),
                                    )

        default_param_set = ParameterSet("Test", values={'a': 1})
        self.parameter_set_list = ParameterSetList(table_kwargs=dict(id=f'parameter-set-selector-'
                                                                        f'{self._instance_index}',
                                                                     columns=[{'name': 'name', 'id': 'name'},
                                                                              {'name': 'values', 'id': 'values'},
                                                                              {'name': 'map_mask', 'id': 'map_mask'},
                                                                              {'name': 'anchor_points', 'id': 'anchor_points'},
                                                                              {'name': 'regions', 'id': 'regions'}],
                                                                     data=kwargs.get('parameter_sets', [{'name': default_param_set.name}]),
                                                                     row_deletable=True,
                                                                     row_selectable='single',))

        tabs = [
            dbc.Tab(label="Values",
                    tab_id=f'parameter-set-values-tab',
                    label_style={'padding': '0.5rem 1rem'}),
            dbc.Tab(label="Points",
                    tab_id=f'parameter-set-points-tab',
                    label_style={'padding': '0.5rem 1rem'}),
            dbc.Tab(label="Regions",
                    tab_id=f'parameter-set-regions-tab',
                    label_style={'padding': '0.5rem 1rem'})
        ]
        self._parameter_set_explorer_content = html.Div(id=f'parameter-set-tabs-content-{self._instance_index}')
        self._parameter_set_explorer_tabs = dbc.Tabs(id=f'parameter-set-tabs-{self._instance_index}',
                                                     active_tab=tabs[0].tab_id,
                                                     children=tabs)
        self.parameter_set_explorer = html.Div([
            self._parameter_set_explorer_tabs,
            self._parameter_set_explorer_content,
        ])

    def init_callbacks(self, app):
        super(SpectraBackgroundRemover, self).init_callbacks(app)

        # Change selection mode
        targeted_callback(self.set_mode,
                          Input(self.selection_mode.id, 'value'),
                          Output(self.id, 'figure'),  # not sure what to output to here; self is convenient?
                          State(self.id, 'figure'),
                          app=app)

        # when plot is clicked, also update region list
        targeted_callback(self._update_region_list,
                          Input(self.id, 'clickData'),
                          Output(self.region_list.data_table.id, 'data'),
                          app=app)

        # When the parameter set tabs are changed, update what parameter set data is shown
        targeted_callback(self._update_parameter_set_explorer_content,
                          Input(self._parameter_set_explorer_tabs.id, 'active_tab'),
                          Output(self._parameter_set_explorer_content.id, 'children'),
                          State(self.parameter_set_list.data_table.id, 'data'),
                          State(self.parameter_set_list.data_table.id, 'selected_rows'),
                          app=app)

        # When the active parameter set changes, update figures and update the config panel content
        targeted_callback(self._active_parameter_set_changed,
                          Input(self.parameter_set_list.data_table.id, 'selected_rows'),
                          Output(self._parameter_set_explorer_content.id, 'children'),
                          State(self.parameter_set_list.data_table.id, 'data'),
                          State(self._parameter_set_explorer_tabs.id, 'active_tab'),
                          app=app)

    def set_mode(self, value):
        self.mode = value

        return self._update_figure()

    def plot_click(self, click_data):
        if self.mode == 1:
            return self._update_energy_line(click_data)
        elif self.mode == 2:
            return self._add_anchor_points(click_data)
        elif self.mode == 3:
            return self._add_anchor_region(click_data)
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

    def _add_anchor_region(self, click_data):
        x = click_data['points'][0]['x']

        # look for a '_region_start' shape already in the figure indicated the previously clicked position
        region_started = next(filter(lambda shape: shape.name=='_region_start', self.figure.layout.shapes), None)

        if region_started:
            region = sorted([region_started.x0, x])
            self.figure.add_vrect(*region, line_width=0, opacity=.3, fillcolor='gray')
            shapes = list(self.figure.layout.shapes)
            shapes.remove(region_started)
            self.figure.layout.shapes = shapes
            raise PreventUpdate
        else:
            self.figure.add_vline(x, name='_region_start')
            self.region_list.data_table.data += [{'name': 'New Region #'}]
            # return self.region_list.data_table.data

        # self.figure.add_vrect(*region_range, line_width=0, opacity=.3, fillcolor='grey')

        return self._update_figure()

    def _update_region_list(self, clickData):
        return self.region_list.data_table.data

    # TODO: set up callbacks (more outputs) where changing the set updates view figures
    def _active_parameter_set_changed(self, selected_rows: List[int], parameter_set_data, active_tab):
        # Update what is shown in views (points, regions, mask) and in the config panel
        ...
        self._update_parameter_set_explorer_content(active_tab, parameter_set_data, selected_rows)

    def _update_parameter_set_explorer_content(self, active_tab, parameter_set_data, selected_rows):
        # Needs to display data in the parameter set corresponding to the tab being switched to
        print('update content')
        if not parameter_set_data:
            print('\tno parameter set data???')
            return

        row = selected_rows[0]  # parameter set list should only have 'single' selection
        print(f'\trow: {row}')
        parameter_set = list(filter(lambda ps: ps['id'] == row, parameter_set_data))[0]  # type: ParameterSet
        print(f'\tset: {parameter_set}')
        if 'values' in active_tab:
            return parameter_set.values
        elif 'points' in active_tab:
            return parameter_set.anchor_points
        elif 'regions' in active_tab:
            return parameter_set.regions

    @property
    def configuration_panel(self):
        children = [
            # dbc.Form(dbc.FormGroup([self.selection_mode])),
            # self.region_list,
            dbc.Label("Parameter Sets"),
            self.parameter_set_list,
            self.parameter_set_explorer,
        ]
        return 'Background Isolator', children

    # def my_background(self, fixed_points, mask, data, ):
