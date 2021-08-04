from itertools import count

from ryujin.components.datalist import DataList


class RegionList(DataList):
    def __init__(self, *args, **kwargs):
        self.region_counter = count(0)

        # Add one initial record
        kwargs['data'] = kwargs.get('data', []) + [{f'Region #{next(self.region_counter)}'}]
        kwargs['active_cell'] = {'column': }

        super(RegionList, self).__init__(*args, **kwargs)