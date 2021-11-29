from .slice import SliceGraph


class MetricsGraph(SliceGraph):
    title = 'Quality'

    def __init__(self, *args, **kwargs):
        super(MetricsGraph, self).__init__(*args, **kwargs)

    def _id(self, instance_index):
        _id = super(MetricsGraph, self)._id(instance_index)
        _id['subtype'] = 'metrics'
        return _id
