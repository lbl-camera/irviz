import enum
import inspect
import os
import tempfile
from functools import cached_property, partial

import dash
import dash_bootstrap_components as dbc
import diskcache
import h5py as h5
import numpy as np
from dash import dcc
from dash import html, Input, State
from dash._utils import create_callback_id, stringify_id
from dash.exceptions import PreventUpdate
from dash.long_callback import DiskcacheLongCallbackManager

from irviz.components.datalists import ParameterSetList, RegionList
from irviz.components.kwarg_editor import StackedKwargsEditor, regularize_name
from irviz.graphs import DecompositionGraph, PairPlotGraph, MapGraph
from irviz.graphs.metrics import MetricsGraph
from irviz.graphs.pair_plot_3D import PairPlot3DGraph
from irviz.graphs.region_spectra_plot import RegionSpectraPlot
from irviz.graphs.slice import SliceGraph
from ryujin.components import Panel, Output
from ryujin.display import ComposableDisplay
from ryujin.utils import targeted_callback

__all__ = ['DecompositionTuner']

tempdir = tempfile.mkdtemp()
cache_path = os.path.join(tempdir, 'cache.h5')


def empty_callable():
    pass


def cache_data(key, data, cache_path):
    print('cache_path:', cache_path)
    with h5.File(cache_path, 'a') as h:
        if key in h:
            del h[key]
        h.create_dataset(key, data=data)


def load_cache(key, cache_path):
    with h5.File(cache_path, 'r') as h:
        return h[key][:]


def filter_values(prefix_name: str, values: dict):
    regularized_name = regularize_name(prefix_name)
    return {name[len(regularized_name)+1:]: value for name, value in values.items() if
            name.startswith(regularized_name)}


def map_enum_values(values: dict, func):
    values = values.copy()
    parameters = inspect.signature(func).parameters
    for param_name in parameters:
        if isinstance(parameters[param_name].default, enum.Enum) and param_name in values:
            values[param_name] = type(parameters[param_name].default).__members__[values[param_name]].value
    return values


class DecompositionParameterSetList(ParameterSetList):
    record_template = {'name': None,
                       'decomposition_function': None,
                       'decomposition_values': dict(),
                       'clustering_function': None,
                       'clustering_values': dict(),
                       'map_mask': None,
                       'selected_regions': []}

    def duplicate_filter(self, key):  # keys of the record template that are duplicated from the active record rather than the template
        return key in ['decomposition_function', 'decomposition_values', 'clustering_function', 'clustering_values']


class TunerPanel(Panel):
    def __init__(self, instance_index, decomposition_funcs, clustering_funcs):
        self.selection_mode = dbc.RadioItems(id=f'background-selection-mode-{instance_index}',
                                             className='btn-group radio-group',
                                             labelClassName='btn btn-secondary',
                                             labelCheckedClassName='active',
                                             options=[{'label': 'Standard Mode', 'value': 1},
                                                      # {'label': 'Fixed Point Mode', 'value': 2},
                                                      {'label': 'Fixed Region Mode', 'value': 2}],
                                             value=1
                                             )

        self.decomposition_kwargs_editor = StackedKwargsEditor(instance_index, decomposition_funcs, 'Decomposition Function',
                                                               id='decomposition-func-selector', )
        self.clustering_kwargs_editor = StackedKwargsEditor(instance_index, clustering_funcs, 'Cluster Function',
                                                            id='clustering-func-selector')
        self.region_list = RegionList(table_kwargs=dict(id=dict(type='region-list', index=instance_index)), )

        # update record template based on func signatures
        record_template = DecompositionParameterSetList.record_template
        decomposition_values = self.decomposition_kwargs_editor.parameter_editor.values
        clustering_values = self.clustering_kwargs_editor.parameter_editor.values
        record_template['decomposition_values'] = decomposition_values
        record_template['clustering_values'] = clustering_values
        record_template['decomposition_function'] = next(iter(decomposition_funcs.keys()))
        record_template['clustering_function'] = next(iter(clustering_funcs.keys()))

        self.parameter_set_list = DecompositionParameterSetList(table_kwargs=dict(
            id=dict(
                type='parameter-set-selector',
                index=instance_index)),
            record_template=record_template)

        self.decomposition_execute = dbc.Button('Execute', id='decomposition-execute')
        self.clustering_execute = dbc.Button('Execute', id='clustering-execute')

        self.decomposition_status = dcc.Store(id=dict(type='decomposition-status', pattern=True),
                                              data='')  # convert to dcc.store later
        self.clustering_status = dcc.Store(id=dict(type='clustering-status', pattern=True), data=False)

        self.cache_path = dcc.Store('cache-path', data=cache_path)

        self._values_tab = dbc.Tab(label="Parameters",
                                   tab_id=f'parameter-set-values-tab',
                                   label_style={'padding': '0.5rem 1rem', },
                                   children=[self.decomposition_kwargs_editor,
                                             self.decomposition_execute,
                                             self.decomposition_status,
                                             html.Hr(),
                                             self.clustering_kwargs_editor,
                                             self.clustering_execute,
                                             self.clustering_status])
        self._regions_tab = dbc.Tab(label="Regions",
                                    tab_id=f'parameter-set-regions-tab',
                                    label_style={'padding': '0.5rem 1rem', },
                                    children=[self.region_list])

        tab_items = [self._values_tab, self._regions_tab]

        tabs = dbc.Tabs(id=f'parameter-set-tabs-{instance_index}',
                        active_tab=tab_items[0].tab_id,
                        children=tab_items)

        print('params:', self.parameter_set_list.data_table.data)

        children = [self.selection_mode,
                    self.parameter_set_list,
                    html.Hr(),
                    tabs,
                    self.cache_path
                    ]

        super(TunerPanel, self).__init__('Decomposition Tuner', children)

    def init_callbacks(self, app):
        super(TunerPanel, self).init_callbacks(app)
        self.parameter_set_list.init_callbacks(app)
        self.decomposition_kwargs_editor.init_callbacks(app)
        self.clustering_kwargs_editor.init_callbacks(app)

        # update parameter set stash when values change
        targeted_callback(partial(self._stash_parameter_set_data, key='selected_regions'),
                          Input(self.region_list.data_table.id, 'data'),
                          Output(self.parameter_set_list.data_table.id, 'data'),
                          State(self.parameter_set_list.data_table.id, 'data'),
                          State(self.parameter_set_list.data_table.id, 'selected_rows'),
                          app=app)
        targeted_callback(
            partial(self._stash_values, key='decomposition_values', editor=self.decomposition_kwargs_editor.parameter_editor),
            Input(self.decomposition_kwargs_editor.parameter_editor.id, 'n_submit'),
            Output(self.parameter_set_list.data_table.id, 'data'),
            State(self.parameter_set_list.data_table.id, 'data'),
            State(self.parameter_set_list.data_table.id, 'selected_rows'),
            app=app)
        targeted_callback(
            partial(self._stash_values, key='clustering_values', editor=self.clustering_kwargs_editor.parameter_editor),
            Input(self.clustering_kwargs_editor.parameter_editor.id, 'n_submit'),
            Output(self.parameter_set_list.data_table.id, 'data'),
            State(self.parameter_set_list.data_table.id, 'data'),
            State(self.parameter_set_list.data_table.id, 'selected_rows'),
            app=app)
        targeted_callback(partial(self._stash_parameter_set_data, key='decomposition_function'),
                          Input(self.decomposition_kwargs_editor.func_selector.id, 'value'),
                          Output(self.parameter_set_list.data_table.id, 'data'),
                          State(self.parameter_set_list.data_table.id, 'data'),
                          State(self.parameter_set_list.data_table.id, 'selected_rows'),
                          app=app)
        targeted_callback(partial(self._stash_parameter_set_data, key='clustering_function'),
                          Input(self.clustering_kwargs_editor.func_selector.id, 'value'),
                          Output(self.parameter_set_list.data_table.id, 'data'),
                          State(self.parameter_set_list.data_table.id, 'data'),
                          State(self.parameter_set_list.data_table.id, 'selected_rows'),
                          app=app)

        # Chained from above; update data lists when the parameter set list is updated
        targeted_callback(self._update_decomposition_values,
                          Input(self.parameter_set_list.data_table.id, 'selected_rows'),
                          Output(self.decomposition_kwargs_editor.parameter_editor.id, 'children'),
                          State(self.parameter_set_list.data_table.id, 'data'),
                          app=app)
        targeted_callback(self._update_clustering_values,
                          Input(self.parameter_set_list.data_table.id, 'selected_rows'),
                          Output(self.clustering_kwargs_editor.parameter_editor.id, 'children'),
                          State(self.parameter_set_list.data_table.id, 'data'),
                          app=app)
        targeted_callback(self._update_decomposition_function,
                          Input(self.parameter_set_list.data_table.id, 'selected_rows'),
                          Output(self.decomposition_kwargs_editor.func_selector.id, 'value'),
                          State(self.parameter_set_list.data_table.id, 'data'),
                          app=app)
        targeted_callback(self._update_clustering_function,
                          Input(self.parameter_set_list.data_table.id, 'selected_rows'),
                          Output(self.clustering_kwargs_editor.func_selector.id, 'value'),
                          State(self.parameter_set_list.data_table.id, 'data'),
                          app=app)
        targeted_callback(self._update_regions_list,
                          Input(self.parameter_set_list.data_table.id, 'selected_rows'),
                          Output(self.region_list.data_table.id, 'data'),
                          State(self.parameter_set_list.data_table.id, 'data'),
                          app=app)

    # def _stash_decomposition_func(self, func_name):
    #     self._stash_parameter_set_data(func_name, 'decomposition_function')
    #     self.parameter_set_list.record_template.update(self.decomposition_kwargs_editor.funcs[func_name])

    def _stash_parameter_set_data(self, data, key):
        _id = create_callback_id(State(self.parameter_set_list.data_table.id, 'data'))
        parameter_set_list = dash.callback_context.states[_id] or []
        current_index = self._find_selected_parameter_set(parameter_set_list)[0]
        parameter_set_list[current_index][key] = data
        return parameter_set_list

    def _stash_values(self, n_submits, key, editor):
        return self._stash_parameter_set_data(editor.values, key)

    def _find_selected_parameter_set(self, parameter_set_list_data=None, row=None) -> (int, dict):
        if parameter_set_list_data is None:
            _id = create_callback_id(State(self.parameter_set_list.data_table.id, 'data'))
            parameter_set_list_data = dash.callback_context.states[_id] or []
        _id = create_callback_id(State(self.parameter_set_list.data_table.id, 'selected_rows'))
        if row is None and _id in dash.callback_context.states:
            row = next(iter(dash.callback_context.states[_id]), None)
        if row is None or row > len(parameter_set_list_data) - 1: raise PreventUpdate
        return row, parameter_set_list_data[row]

    def _update_decomposition_values(self, selected_rows):
        values = self._find_selected_parameter_set(row=next(iter(selected_rows), None))[-1]['decomposition_values']
        # function = self._find_selected_parameter_set(row=next(iter(selected_rows), None))[-1]['decomposition_function']
        # rebuild the child items
        return self.decomposition_kwargs_editor.parameter_editor.build_children(values)

    def _update_clustering_values(self, selected_rows):
        values = self._find_selected_parameter_set(row=next(iter(selected_rows), None))[-1]['clustering_values']
        # function = self._find_selected_parameter_set(row=next(iter(selected_rows), None))[-1]['decomposition_function']
        # rebuild the child items
        return self.clustering_kwargs_editor.parameter_editor.build_children(values)

    def _update_decomposition_function(self, selected_rows):
        return self._find_selected_parameter_set(row=next(iter(selected_rows), None))[-1]['decomposition_function']

    def _update_clustering_function(self, selected_rows):
        return self._find_selected_parameter_set(row=next(iter(selected_rows), None))[-1]['clustering_function']

    def _update_regions_list(self, selected_rows):
        return self._find_selected_parameter_set(row=next(iter(selected_rows), None))[-1]['selected_regions']


cache = diskcache.Cache(tempdir)


class DecompositionTuner(ComposableDisplay):
    _precision = 2
    plugins = [
        # dl.plugins.FlexibleCallbacks(),
        # dl.plugins.HiddenComponents(),
        # dl.plugins.LongCallback(long_callback_manager)
    ]
    long_callback_manager = DiskcacheLongCallbackManager(cache)

    def init_components(self, decomposition_functions, *args, cluster_functions=None, static_mask=None, **kwargs):
        self.decomposition_functions = decomposition_functions
        self.cluster_functions = cluster_functions
        self.data = kwargs.get('data')
        self.bounds = kwargs.get('bounds')
        self._static_mask = static_mask

        components = super(DecompositionTuner, self).init_components(*args, **kwargs)

        style = dict(display='flex',
                     flexDirection='row',
                     height='100%',
                     minHeight='450px')
        graph_kwargs = dict(style=style, responsive=True)

        if static_mask is None:
            static_mask = np.ones_like(self.data[0]) * np.NaN
        static_mask = static_mask.astype('O')
        static_mask[np.logical_not(static_mask.astype(np.bool_))] = np.NaN
        static_mask[static_mask == 1] = 1  # casts True -> 1
        static_mask_trace = SliceGraph._get_image_trace(static_mask,
                                                        self.bounds,
                                                        colorscale='reds',
                                                        opacity=0.3,
                                                        showscale=False,
                                                        hoverinfo='skip',
                                                        name='static mask', )

        map_graph = MapGraph(instance_index=self._instance_index,
                             graph_kwargs={**graph_kwargs, 'className': 'col-lg-4 p-0'},
                             traces=[static_mask_trace],
                             **kwargs)
        self.spectra_graph = RegionSpectraPlot(instance_index=self._instance_index,
                                               # background_func=background_function,
                                               graph_kwargs={**graph_kwargs, 'className': 'col-lg-8 p-0'},
                                               **kwargs)
        kwargs['data'] = np.zeros((1, *kwargs['data'].shape[1:]))
        self.decomposition_graph = DecompositionGraph(instance_index=self._instance_index,
                                                      graph_kwargs={**graph_kwargs, 'className': 'col-lg-4 p-0'},
                                                      cluster_labels=None,
                                                      cluster_label_names=None,
                                                      traces=[static_mask_trace],
                                                      **kwargs)
        self.pair_plot = PairPlotGraph(instance_index=self._instance_index,
                                       graph_kwargs={**graph_kwargs, 'className': 'col-lg-4 p-0'},
                                       cluster_labels=None,
                                       cluster_label_names=None,
                                       **kwargs)
        self.pair_plot_3D = PairPlot3DGraph(instance_index=self._instance_index,
                                            graph_kwargs={**graph_kwargs, 'className': 'col-lg-4 p-0'},
                                            cluster_labels=None,
                                            cluster_label_names=None,
                                            **kwargs)
        self.quality_graph = MetricsGraph(instance_index=self._instance_index,
                                          graph_kwargs={**graph_kwargs, 'className': 'col-lg-4 p-0'},
                                          cluster_labels=None,
                                          cluster_label_names=None,
                                          traces=[static_mask_trace],
                                          **kwargs)
        self._static_mask_stash = dcc.Store(id=dict(type='static-mask', index='instance_index'))
        components.extend(
            [map_graph, self.decomposition_graph, self.quality_graph, self.spectra_graph, self.pair_plot, self.pair_plot_3D,
             self._static_mask_stash])
        # components.append(BackgroundMapGraph(instance_index=self._instance_index,
        #                                      graph_kwargs=graph_kwargs,
        #                                      # mask=mask,
        #                                      **kwargs))
        # components.append(PairPlotGraph(instance_index=self._instance_index,
        #                                 graph_kwargs=graph_kwargs,
        #                                 cluster_labels=None,
        #                                 cluster_label_names=[],
        #                                 **kwargs))

        # construct extra panel
        self.tuner_panel = TunerPanel(self._instance_index, decomposition_functions, cluster_functions)

        return components

    def init_callbacks(self, app):
        super(DecompositionTuner, self).init_callbacks(app)

        # callback sequence:
        # 0. serialize data to disk (long)
        # 1. Status change -> execute decomposition, overwrite serialized data (long) -> emit path to status 2
        # 2. Status 2 change -> load lazy data object, pass to displays -> displays

        # Ok, so there's a bit of a trick here. Unfortunately long_callbacks are broken for partials, as the callback
        # must be dill pickleable; the next lines are necessary to extract the functions dict out from the state of the Tuner
        # instance into dill-compatible local
        decomposition_funcs = self.decomposition_functions
        cluster_funcs = self.cluster_functions
        bounds = self.bounds
        map_data = self.data

        def execute_decomposition(n_clicks, data, selected_rows, cache_path, static_mask):
            parameter_set = data[selected_rows[0]]
            decomposition_function = decomposition_funcs[parameter_set['decomposition_function']]
            values = map_enum_values(
                filter_values(parameter_set['decomposition_function'], parameter_set['decomposition_values']),
                decomposition_function)
            print(f'Running {parameter_set["decomposition_function"]} with parameters: {values}')

            if len(bounds[0]) == 2:
                spectral_coords = np.linspace(bounds[0][0], bounds[0][1], map_data.shape[0])
            else:
                spectral_coords = bounds[0]

            decomposition, components, quality = decomposition_function \
                (spectral_coords,
                 map_data,
                 np.logical_and(parameter_set['map_mask'] or np.ones(map_data.shape[1:], dtype=np.bool_),
                                np.logical_not(static_mask or np.zeros(map_data.shape[1:], dtype=np.bool_))),
                 parameter_set['selected_regions'] or [{'region_min': spectral_coords.min(),
                                                        'region_max': spectral_coords.max()}],
                 **values)

            cache_data('decomposition', decomposition, cache_path)
            cache_data('components', components, cache_path)
            cache_data('quality', quality, cache_path)

            return decomposition.shape[0]

        def execute_clustering(n_clicks, data, selected_rows, cache_path, static_mask):
            parameter_set = data[selected_rows[0]]
            values = filter_values(parameter_set['clustering_function'], parameter_set['clustering_values'])
            print(f'Running {parameter_set["clustering_function"]} with parameters: {values}')

            decomposition_data = load_cache('decomposition', cache_path)

            clustering = cluster_funcs[parameter_set['clustering_function']] \
                (decomposition_data,
                 np.logical_and(parameter_set['map_mask'] or np.ones(map_data.shape[1:], dtype=np.bool_),
                                np.logical_not(static_mask or np.zeros(map_data.shape[1:], dtype=np.bool_))),
                 **values)

            cache_data('clustering', clustering, cache_path)

            return True

        # Decomposition processing (background)
        app.long_callback(
            Output(self.tuner_panel.decomposition_status.id, "data"),
            Input(self.tuner_panel.decomposition_execute.id, "n_clicks"),
            State(self.tuner_panel.parameter_set_list.data_table.id, 'data'),
            State(self.tuner_panel.parameter_set_list.data_table.id, 'selected_rows'),
            State(self.tuner_panel.cache_path.id, 'data'),
            State(self._static_mask_stash.id, 'data'),
            running=[
                (Output(self.tuner_panel.decomposition_execute.id, "disabled"), True, False),
            ],
            prevent_initial_call=True
        )(execute_decomposition)

        # Clustering processing (background)
        app.long_callback(
            Output(self.tuner_panel.clustering_status.id, "data"),
            Input(self.tuner_panel.clustering_execute.id, "n_clicks"),
            State(self.tuner_panel.parameter_set_list.data_table.id, 'data'),
            State(self.tuner_panel.parameter_set_list.data_table.id, 'selected_rows'),
            State(self.tuner_panel.cache_path.id, 'data'),
            State(self._static_mask_stash.id, 'data'),
            running=[
                (Output(self.tuner_panel.clustering_execute.id, "disabled"), True, False),
            ],
            prevent_initial_call=True
        )(execute_clustering)

        # chained call returning from decomposition execute; displays result
        targeted_callback(self.update_decomposition,
                          Input(self.tuner_panel.decomposition_status.id, 'data'),
                          Output(self.decomposition_graph.id, 'figure'),
                          State(self.tuner_panel.cache_path.id, 'data'),
                          app=app)
        targeted_callback(self.update_pair_plot_decomposition,
                          Input(self.tuner_panel.decomposition_status.id, 'data'),
                          Output(self.pair_plot.id, 'figure'),
                          State(self.tuner_panel.cache_path.id, 'data'),
                          State(self.pair_plot.configuration_panel._decomposition_component_1.id, 'value'),
                          State(self.pair_plot.configuration_panel._decomposition_component_2.id, 'value'),
                          app=app)
        targeted_callback(self.update_3D_pair_plot_decomposition,
                          Input(self.tuner_panel.decomposition_status.id, 'data'),
                          Output(self.pair_plot_3D.id, 'figure'),
                          State(self.tuner_panel.cache_path.id, 'data'),
                          State(self.pair_plot_3D.configuration_panel._decomposition_component_1.id, 'value'),
                          State(self.pair_plot_3D.configuration_panel._decomposition_component_2.id, 'value'),
                          State(self.pair_plot_3D.configuration_panel._decomposition_component_3.id, 'value'),
                          app=app)
        targeted_callback(self.update_quality,
                          Input(self.tuner_panel.decomposition_status.id, 'data'),
                          Output(self.quality_graph.id, 'figure'),
                          State(self.tuner_panel.cache_path.id, 'data'),
                          app=app)

        # chained call returning from clustering execute; displays result
        targeted_callback(self.update_clustering,
                          Input(self.tuner_panel.clustering_status.id, 'data'),
                          Output(self.decomposition_graph.id, 'figure'),
                          State(self.tuner_panel.cache_path.id, 'data'),
                          app=app)
        targeted_callback(self.update_pair_plot_clustering,
                          Input(self.tuner_panel.clustering_status.id, 'data'),
                          Output(self.pair_plot.id, 'figure'),
                          State(self.tuner_panel.cache_path.id, 'data'),
                          State(self.pair_plot.configuration_panel._decomposition_component_1.id, 'value'),
                          State(self.pair_plot.configuration_panel._decomposition_component_2.id, 'value'),
                          app=app)
        targeted_callback(self.update_pair_plot_3D_clustering,
                          Input(self.tuner_panel.clustering_status.id, 'data'),
                          Output(self.pair_plot_3D.id, 'figure'),
                          State(self.tuner_panel.cache_path.id, 'data'),
                          State(self.pair_plot_3D.configuration_panel._decomposition_component_1.id, 'value'),
                          State(self.pair_plot_3D.configuration_panel._decomposition_component_2.id, 'value'),
                          State(self.pair_plot_3D.configuration_panel._decomposition_component_3.id, 'value'),
                          app=app)

        # update parameter set anchor region
        targeted_callback(self._add_selected_region,
                          Input(self.spectra_graph.id, 'clickData'),
                          Output(self.tuner_panel.region_list.data_table.id, 'data'),
                          State(self.tuner_panel.region_list.data_table.id, 'data'),
                          State(self.tuner_panel.selection_mode.id, 'value'),
                          app=app)

        # update plot figure on parameter set change
        targeted_callback(self.update_figure_on_data_change,
                          Input(self.tuner_panel.parameter_set_list.data_table.id, 'data'),
                          Output(self.spectra_graph.id, 'figure'),
                          State(self.tuner_panel.parameter_set_list.data_table.id, 'selected_rows'),
                          # State(dict(type='slice_graph',
                          #            subtype='background-map',
                          #            index=self._instance_index),
                          #       'figure'),
                          app=app)

    def make_layout(self):
        return html.Div(html.Div(self.components, className='row'), className='container-fluid')  # , style={'flexGrow': 1})

    @property
    def parameter_sets(self):
        return self.components[0]._parameter_sets

    @cached_property
    def panels(self):
        return [self.tuner_panel] + super(DecompositionTuner, self).panels

    def update_decomposition(self, status):
        cache_path = dash.callback_context.states[self.tuner_panel.cache_path.id + '.data']
        data = load_cache('decomposition', cache_path)
        figure = self.decomposition_graph.rebuild_component_heatmaps(data)
        return figure

    def update_pair_plot_decomposition(self, status):
        cache_path = dash.callback_context.states[self.tuner_panel.cache_path.id + '.data']
        data = load_cache('decomposition', cache_path)
        self.pair_plot.set_data(data)
        return self.pair_plot.show_pair_plot()

    def update_pair_plot_clustering(self, status):
        cache_path = dash.callback_context.states[self.tuner_panel.cache_path.id + '.data']
        clustering = load_cache('clustering', cache_path)
        self.pair_plot.set_clustering(clustering)
        return self.pair_plot.show_pair_plot()

    def update_3D_pair_plot_decomposition(self, status):
        cache_path = dash.callback_context.states[self.tuner_panel.cache_path.id + '.data']
        data = load_cache('decomposition', cache_path)
        self.pair_plot_3D.set_data(data)
        return self.pair_plot_3D.show_pair_plot()

    def update_pair_plot_3D_clustering(self, status):
        cache_path = dash.callback_context.states[self.tuner_panel.cache_path.id + '.data']
        clustering = load_cache('clustering', cache_path)
        self.pair_plot_3D.set_clustering(clustering)
        return self.pair_plot_3D.show_pair_plot()

    def update_clustering(self, status):
        cache_path = dash.callback_context.states[self.tuner_panel.cache_path.id + '.data']
        data = load_cache('clustering', cache_path)
        figure = self.decomposition_graph.set_clustering(data)
        return figure

    def update_quality(self, status):
        cache_path = dash.callback_context.states[self.tuner_panel.cache_path.id + '.data']
        data = np.expand_dims(load_cache('quality', cache_path), 0)
        figure = self.quality_graph.set_data(data)
        return figure

    def _add_selected_region(self, click_data):
        mode = dash.callback_context.states[f'{self.tuner_panel.selection_mode.id}.value'] or []
        if not mode == 2:
            raise PreventUpdate

        x = click_data['points'][0]['x']

        data = dash.callback_context.states[f'{stringify_id(self.tuner_panel.region_list.data_table.id)}.data'] or []

        # look for a '_region_start' shape already in the figure indicated the previously clicked position
        last_region = data[-1] if len(data) else {}

        if 'region_max' in last_region and last_region['region_max'] is None:
            region = sorted([last_region['region_min'], x])

            data[-1]['region_min'] = round(region[0], self._precision)
            data[-1]['region_max'] = round(region[1], self._precision)
        else:
            data += [
                {'name': f'Region #{next(self.tuner_panel.region_list.region_counter)}',
                 'region_min': round(x, self._precision),
                 'region_max': None}]
        return data

    def update_figure_on_data_change(self, data):
        # Also, stash the new data for property access
        self._parameter_sets = data
        parameter_set = self.tuner_panel._find_selected_parameter_set(parameter_set_list_data=data)[-1]
        return self.spectra_graph._update_regions(parameter_set['selected_regions'])
