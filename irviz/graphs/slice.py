from itertools import count

import dash_core_components as dcc
from plotly import graph_objects as go


class SliceGraph(dcc.Graph):
    """Dash Graph for viewing 2D slices of 3D data.

    Parameters
    ----------
    data : dask.array
        3D data array
    parent : html.Div
        The parent object that creates this Graph

    """
    _counter = count(0)

    title = ''
    aspect_locked = True

    def __init__(self, data, bounds, parent, slice_axis=0, traces=None, shapes=None, **kwargs):

        # Cache our data and parent for use in the callbacks
        self._data = data
        self._parent = parent
        self._bounds = bounds
        self._instance_index = next(self._counter)
        self._traces = traces or []
        self._shapes = shapes or []

        self.xaxis_title = kwargs.pop('xaxis_title', '')
        self.yaxis_title = kwargs.pop('yaxis_title', '')

        self._h_line = go.layout.Shape(type='line',
                                       # width=3,
                                       xref='paper',
                                       yref='y',
                                       x0=0,
                                       x1=1,
                                       y0=(bounds[1][1]+bounds[1][0])/2,
                                       y1=(bounds[1][1]+bounds[1][0])/2)
        self._v_line = go.layout.Shape(type='line',
                                       xref='x',
                                       yref='paper',
                                       x0=(bounds[2][1]+bounds[2][0])/2,
                                       x1=(bounds[2][1]+bounds[2][0])/2,
                                       y0=0,
                                       y1=1)

        self._shapes.extend([self._h_line, self._v_line])

        figure = self._update_figure()
        super(SliceGraph, self).__init__(figure=figure,
                                         id=self._id(),
                                         className='col-lg-3',
                                         responsive=True,
                                         style=dict(display='flex',
                                                    flexDirection='row',
                                                    height='100%',
                                                    minHeight='450px'),
                                         **kwargs)

    def _id(self):
        return f'slicegraph_{self._instance_index}'

    def register_callbacks(self):
        ...

    def _update_figure(self):
        fig = go.Figure(self._traces)
        fig.update_layout(title=self.title,
                          xaxis_title=self.xaxis_title,
                          yaxis_title=self.yaxis_title)
        if self.aspect_locked:
            fig.update_yaxes(scaleanchor="x", scaleratio=1)
        for shape in self._shapes:
            fig.add_shape(shape)
        return fig

    def show_click(self, click_data):
        y_index = click_data["points"][0]["y"]
        x_index = click_data["points"][0]["x"]
        self._h_line.y0 = y_index
        self._h_line.y1 = y_index
        self._v_line.x0 = x_index
        self._v_line.x1 = x_index

        return self._update_figure()

    @property
    def position(self):
        """The current spatial position of the crosshair"""
        return self._v_line.x0, self._h_line.y0