from dataclasses import dataclass
from itertools import count

import numpy as np

from ryujin.components.datalist import DataList


@dataclass
class ParameterSet:
    name: str
    values: dict = dict(),
    map_mask: np.ndarray = None,
    anchor_points: list = None
    regions: list = None


class RegionList(DataList):
    def __init__(self, *args, **kwargs):
        self.region_counter = count(0)

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
    def __init__(self, *args, table_kwargs=None, **kwargs):
        self.parameter_set_counter = count(0)
        # TODO: once API established, update the kwargs being searched

        # Always start with on record
        table_kwargs = table_kwargs or {}
        table_kwargs['data'] = table_kwargs.get('data', [])
        if not table_kwargs['data']:
            table_kwargs['data'] += [{'name': f'Parameter Set #{next(self.parameter_set_counter)}',
                                      'parameter_set': ParameterSet()}]

        super(ParameterSetList, self).__init__(*args, **kwargs)
