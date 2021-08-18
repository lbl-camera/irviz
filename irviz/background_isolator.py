import dash_html_components as html
import dash_core_components as dcc

from irviz.graphs import MapGraph
from irviz.graphs.background_map import BackgroundMapGraph
from irviz.graphs.spectra_background_remover import SpectraBackgroundRemover
from ryujin.display import ComposableDisplay

__all__ = ['BackgroundIsolator']


def empty_callable():
    pass


class BackgroundIsolator(ComposableDisplay):

    def init_components(self, *args, background_function=empty_callable, **kwargs):
        components = super(BackgroundIsolator, self).init_components(*args, **kwargs)

        style = dict(display='flex',
                     flexDirection='row',
                     height='100%',
                     minHeight='450px')
        className = 'col-lg-12 p-0'
        graph_kwargs = dict(style=style, className=className, responsive=True)

        components.append(SpectraBackgroundRemover(instance_index=self._instance_index,
                                                   background_func=background_function,
                                                   graph_kwargs=graph_kwargs,
                                                   **kwargs))
        components.append(BackgroundMapGraph(instance_index=self._instance_index,
                                             graph_kwargs=graph_kwargs,
                                             **kwargs))
        components.append(dcc.Interval('background-update', interval=1*1000))

        return components

    def make_layout(self):
        return html.Div(html.Div(self.components, className='row'), className='container-fluid') #, style={'flexGrow': 1})

    @property
    def parameter_sets(self):
        return self.components[0]._parameter_sets
