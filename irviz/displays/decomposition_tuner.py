import tempfile
from functools import cached_property, partial
import os

import diskcache
import dash
from dash import html, Input, State
import dash_bootstrap_components as dbc
from dash import dcc
import numpy as np
from dash._utils import create_callback_id
from dash.exceptions import PreventUpdate
from dash.long_callback import DiskcacheLongCallbackManager
import h5py as h5

from irviz.components.datalists import BackgroundIsolatorParameterSetList, ParameterSetList
from irviz.components.kwarg_editor import StackedKwargsEditor, regularize_name
from irviz.graphs import SpectraPlotGraph, DecompositionGraph, PairPlotGraph, MapGraph
from irviz.graphs.background_map import BackgroundMapGraph
from irviz.graphs.slice import SliceGraph
from irviz.graphs.spectra_background_remover import SpectraBackgroundRemover
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
            h[key][...] = data
        else:
            h.create_dataset(key, data=data)


def load_cache(key, cache_path):
    h = h5.File(cache_path, 'a')
    return h[key]


def filter_values(prefix_name:  str, values:dict):
    regularized_name = regularize_name(prefix_name)
    return {name.removeprefix(regularized_name+'-'): value for name, value in values.items() if name.startswith(regularized_name)}


class DecompositionParameterSetList(ParameterSetList):
    record_template = {'name': None,
                       'decomposition_function': None,
                       'decomposition_values': dict(),
                       'clustering_function': None,
                       'clustering_values': dict(),
                       'map_mask': None,
                       'selected_regions': []}


class TunerPanel(Panel):
    def __init__(self, instance_index, decomposition_funcs, clustering_funcs):
        self.parameter_set_list = DecompositionParameterSetList(table_kwargs=dict(
            id=dict(
                type='parameter-set-selector',
                index=instance_index)))

        self.decomposition_kwargs_editor = StackedKwargsEditor(instance_index, decomposition_funcs, 'Decomposition Function', id='decomposition-func-selector',)
        self.clustering_kwargs_editor = StackedKwargsEditor(instance_index, clustering_funcs, 'Cluster Function', id='clustering-func-selector')

        self.decomposition_execute = dbc.Button('Execute', id='decomposition-execute')
        self.clustering_execute = dbc.Button('Execute', id='clustering-execute')

        self.decomposition_status = dcc.Store(id=dict(type='decomposition-status', pattern=True), data='')  # convert to dcc.store later
        self.clustering_status = dcc.Store(id=dict(type='clustering-status', pattern=True), data=False)

        self.cache_path = dcc.Store('cache-path', data=cache_path)

        children = [self.parameter_set_list,
                    html.Hr(),
                    self.decomposition_kwargs_editor,
                    self.decomposition_execute,
                    self.decomposition_status,
                    html.Hr(),
                    self.clustering_kwargs_editor,
                    self.clustering_execute,
                    self.clustering_status,
                    self.cache_path
                    ]

        super(TunerPanel, self).__init__('Decomposition Tuner', children)

    def init_callbacks(self, app):
        super(TunerPanel, self).init_callbacks(app)
        self.parameter_set_list.init_callbacks(app)
        self.decomposition_kwargs_editor.init_callbacks(app)
        self.clustering_kwargs_editor.init_callbacks(app)

        targeted_callback(partial(self._stash_values, key='decomposition_values', editor=self.decomposition_kwargs_editor.parameter_editor),
                          Input(self.decomposition_kwargs_editor.parameter_editor.id, 'n_submit'),
                          Output(self.parameter_set_list.data_table.id, 'data'),
                          State(self.parameter_set_list.data_table.id, 'data'),
                          State(self.parameter_set_list.data_table.id, 'selected_rows'),
                          app=app)

        targeted_callback(partial(self._stash_values, key='clustering_values', editor=self.clustering_kwargs_editor.parameter_editor),
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
        if row is None or row > len(parameter_set_list_data)-1: raise PreventUpdate
        return row, parameter_set_list_data[row]


cache = diskcache.Cache(tempdir)


class DecompositionTuner(ComposableDisplay):
    plugins = [
        # dl.plugins.FlexibleCallbacks(),
        # dl.plugins.HiddenComponents(),
        # dl.plugins.LongCallback(long_callback_manager)
    ]
    long_callback_manager = DiskcacheLongCallbackManager(cache)

    def init_components(self, decomposition_functions, *args, cluster_functions=None, quality_functions=None, **kwargs):
        self.decomposition_functions = decomposition_functions
        self.cluster_functions = cluster_functions
        self.quality_functions = quality_functions
        self.data = kwargs.get('data')
        self.bounds = kwargs.get('bounds')

        components = super(DecompositionTuner, self).init_components(*args, **kwargs)

        style = dict(display='flex',
                     flexDirection='row',
                     height='100%',
                     minHeight='450px')
        className = 'col-lg-12 p-0'
        graph_kwargs = dict(style=style, className=className, responsive=True)

        map_graph = MapGraph(instance_index=self._instance_index,
                             graph_kwargs=graph_kwargs,
                             **kwargs)
        spectra_graph = SpectraPlotGraph(instance_index=self._instance_index,
                                         # background_func=background_function,
                                         graph_kwargs=graph_kwargs,
                                         **kwargs)
        kwargs['data'] = np.zeros((1, *kwargs['data'].shape[1:]))
        self.decomposition_graph = DecompositionGraph(instance_index=self._instance_index,
                                                      graph_kwargs=graph_kwargs,
                                                      cluster_labels=None,
                                                      cluster_label_names=None,
                                                      **kwargs)
        components.extend([map_graph, self.decomposition_graph, spectra_graph])
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

        def execute_decomposition(n_clicks, data, selected_rows, cache_path):
            parameter_set = data[selected_rows[0]]
            values = filter_values(parameter_set['decomposition_function'], parameter_set['decomposition_values'])
            print(f'Running {parameter_set["decomposition_function"]} with parameters: {values}')

            if len(bounds[0]) == 2:
                spectral_coords = np.linspace(bounds[0][0], bounds[0][1], map_data.shape[0])
            else:
                spectral_coords = bounds[0]

            decomposition, components = decomposition_funcs[parameter_set['decomposition_function']] \
                (spectral_coords,
                 map_data,
                 parameter_set['map_mask'] or np.ones(map_data.shape[1:], dtype=np.bool_),
                 parameter_set['selected_regions'] or [{'region_min': spectral_coords.min(),
                                                        'region_max': spectral_coords.max()}],
                 **values)

            cache_data('decomposition', decomposition, cache_path)
            cache_data('components', components, cache_path)

            return decomposition.shape[0]

        def execute_clustering(n_clicks, data, selected_rows, cache_path):
            parameter_set = data[selected_rows[0]]
            values = filter_values(parameter_set['clustering_function'], parameter_set['clustering_values'])
            print(f'Running {parameter_set["clustering_function"]} with parameters: {values}')

            decomposition_data = load_cache('decomposition', cache_path)

            clustering = cluster_funcs[parameter_set['clustering_function']] \
                (decomposition_data,
                 parameter_set['map_mask'] or np.ones(map_data.shape[1:], dtype=np.bool_),
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

        # chained call returning from decomposition execute; displays result
        targeted_callback(self.update_clustering,
                          Input(self.tuner_panel.clustering_status.id, 'data'),
                          Output(self.decomposition_graph.id, 'figure'),
                          State(self.tuner_panel.cache_path.id, 'data'),
                          app=app)


    def make_layout(self):
        return html.Div(html.Div(self.components, className='row'), className='container-fluid') #, style={'flexGrow': 1})

    @property
    def parameter_sets(self):
        return self.components[0]._parameter_sets

    @cached_property
    def panels(self):
        return [self.tuner_panel] + super(DecompositionTuner, self).panels

    def update_decomposition(self, status):
        cache_path = dash.callback_context.states[self.tuner_panel.cache_path.id+'.data']
        data = load_cache('decomposition', cache_path)
        figure = self.decomposition_graph.rebuild_component_heatmaps(data)
        return figure

    def update_clustering(self, status):
        cache_path = dash.callback_context.states[self.tuner_panel.cache_path.id+'.data']
        data = load_cache('clustering', cache_path)
        figure = self.decomposition_graph.set_clustering(data)
        return figure