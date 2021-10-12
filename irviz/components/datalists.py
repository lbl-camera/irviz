from itertools import count

import dash
from dash._utils import create_callback_id
from dash.dependencies import Output, Input, State

from ryujin.components.datalist import DataList
from ryujin.utils import targeted_callback


class RegionList(DataList):
    def __init__(self, *args, **kwargs):
        self.region_counter = count(1)

        # Add one initial record
        table_kwargs = kwargs.get('table_kwargs', {})
        # table_kwargs['data'] = table_kwargs.get('data', []) + [{'name': f'Region #{next(self.region_counter)}'}]
        table_kwargs['active_cell'] = {'column': 0, 'row': 0}
        table_kwargs['row_deletable'] = True
        table_kwargs['columns'] = [{'name': 'name', 'id': 'name'},
                                   {'name': 'region_min', 'id': 'region_min'},
                                   {'name': 'region_max', 'id': 'region_max'}]
        table_kwargs['editable'] = True
        kwargs['table_kwargs'] = table_kwargs

        super(RegionList, self).__init__(*args, **kwargs)


class AnchorPointList(DataList):
    def __init__(self, *args, **kwargs):
        self.point_counter = count(1)

        table_kwargs = kwargs.get('table_kwargs', {})
        table_kwargs['row_deletable'] = True
        table_kwargs['columns'] = [{'name': 'name', 'id': 'name'},
                                   {'name': 'x', 'id': 'x'},
                                   # {'name': 'y', 'id': 'y'}
                                   ]
        table_kwargs['editable'] = True
        kwargs['table_kwargs'] = table_kwargs

        super(AnchorPointList, self).__init__(*args, **kwargs)


class ParameterSetList(DataList):
    def __init__(self, *args, table_kwargs=None, **kwargs):
        self.parameter_set_counter = count(1)
        # TODO: once API established, update the kwargs being searched

        # Always start with on record
        table_kwargs = table_kwargs or {}
        table_kwargs['data'] = table_kwargs.get('data', [])
        if not table_kwargs['data']:
            record = self.new_record()
            table_kwargs['data'].append(record)

        table_kwargs['data'][0]['selected'] = True
        table_kwargs['columns'] = [{'name': 'name', 'id': 'name'}]
        table_kwargs['row_deletable'] = True
        table_kwargs['row_selectable'] = 'single'
        table_kwargs['selected_rows'] = [0]

        super(ParameterSetList, self).__init__(*args, table_kwargs=table_kwargs, **kwargs)

    def init_callbacks(self, app):
        targeted_callback(self.assert_selected_row,
                          Input(self.data_table.id, 'data_previous'),
                          Output(self.data_table.id, 'selected_rows'),
                          State(self.data_table.id, 'selected_rows'),
                          State(self.data_table.id, 'data'),
                          app=app)

    def assert_selected_row(self, data):
        _id = create_callback_id(State(self.data_table.id, 'selected_rows'))
        selected_rows = dash.callback_context.states[_id]
        _id = create_callback_id(State(self.data_table.id, 'data'))
        data = dash.callback_context.states[_id]

        if not selected_rows and len(data):
            selected_rows = [0]

        return selected_rows

    def new_record(self):
        name = f'Parameter Set #{next(self.parameter_set_counter)}'
        record = self.record_template.copy()
        record['name'] = name
        return record


class BackgroundIsolatorParameterSetList(ParameterSetList):
    record_template = {'name': None,
                       'values': dict(),
                       'map_mask': None,
                       'anchor_points': [],
                       'anchor_regions': [],
                       'selected': False}


