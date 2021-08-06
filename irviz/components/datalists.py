from dataclasses import dataclass
from itertools import count

import numpy as np

from ryujin.components.datalist import DataList


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
        kwargs['table_kwargs'] = table_kwargs

        super(RegionList, self).__init__(*args, **kwargs)


class ParameterSetList(DataList):
    record_template = {'name': None,
                       'values': dict(),
                       'map_mask': None,
                       'anchor_points': [],
                       'anchor_regions': []}

    def __init__(self, *args, table_kwargs=None, **kwargs):
        self.parameter_set_counter = count(1)
        # TODO: once API established, update the kwargs being searched

        # Always start with on record
        table_kwargs = table_kwargs or {}
        table_kwargs['data'] = table_kwargs.get('data', [])
        if not table_kwargs['data']:
            name = f'Parameter Set #{next(self.parameter_set_counter)}'
            record = self.record_template.copy()
            record['name'] = name
            table_kwargs['data'].append(record)

        table_kwargs['columns'] = [{'name': 'name', 'id': 'name'}]
        table_kwargs['row_deletable'] = True
        table_kwargs['row_selectable'] = 'single'

        super(ParameterSetList, self).__init__(*args, table_kwargs=table_kwargs, **kwargs)
