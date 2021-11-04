from functools import cached_property

from ryujin.components import Panel
from .slice import SliceGraph
import dash_bootstrap_components as dbc


class MetricsGraph(SliceGraph):
    def __init__(self, *args, **kwargs):
        super(MetricsGraph, self).__init__(*args, **kwargs)

    def _id(self, instance_index):
        _id = super(MetricsGraph, self)._id(instance_index)
        _id['subtype'] = 'metrics'
        return _id
