from itertools import count

from ryujin.components.datalist import DataList


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
