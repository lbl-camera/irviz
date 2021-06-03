import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import plotly.graph_objects as go
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate

from irviz.graphs import SliceGraph, SpectraPlotGraph


class Viewer(html.Div):
    _global_slicer_counter = 0

    def __init__(self, app, data, decomposition=None, bounds=None,):
        self.data = data
        self._app = app
        self.decomposition = decomposition
        self.bounds = bounds

        Viewer._global_slicer_counter += 1

        # Initialize graphs
        self.spectra_graph = SpectraPlotGraph(data, self)
        self.slice_graph = SliceGraph(data, self)
        # self.orthogonal_x_graph = SliceGraph(data, self)
        # self.orthogonal_y_graph = SliceGraph(data, self)
        self.decomposition_graph = SliceGraph(self.decomposition, self)
        self.pair_plot_graph = dcc.Graph(id=f'pair_plot_{self._global_slicer_counter}')
        
        # Set up callbacks (Graphs need to wait until all children in this viewer are init'd)
        self.spectra_graph.register_callbacks()
        self.slice_graph.register_callbacks()

        component_selector = html.Div(
            [
                dbc.RadioItems(
                    id='component-selector',
                    className='btn-group',
                    labelClassName="btn btn-secondary",
                    labelCheckedClassName="active",
                    options=[
                        {'label': 'Component 1', 'value': 0},
                        {'label': 'Component 2', 'value': 1},
                        {'label': 'Component 3', 'value': 2}
                    ],
                    value=0),
            ],
            className='radio-group'
        )

        # Initialize layout
        children = html.Div([self.slice_graph,
                             self.spectra_graph,
                             component_selector,
                             self.decomposition_graph,])

        super(Viewer, self).__init__(children=children,
                                     style={'display': 'grid',
                                            'gridTemplateColumns': '50% 50%',
                                            },
                                     )


def notebook_viewer(data, decomposition=None, bounds=None, mode='inline'):
    was_running = True
    import irviz
    try:
        from jupyter_dash import JupyterDash
    except ImportError:
        print("Please install jupyter-dash first.")
    else:
        if not irviz.app:
            # Creating a new app means we never ran the server
            irviz.app = JupyterDash(__name__)
            was_running = False

    app = irviz.app
    viewer = Viewer(app, data.compute(), decomposition, bounds)
    # viewer2 = Viewer(data.compute(), app=app)

    div = html.Div(children=[viewer])#, viewer2])
    app.layout = div

    # Prevent server from being run multiple times; only one instance allowed
    if not was_running:
        irviz.app.run_server(mode=mode)
    else:
        # Values passed here are from
        # jupyter_app.jupyter_dash.JupyterDash.run_server
        app._display_in_jupyter(dashboard_url='http://127.0.0.1:8050/',
                                mode=mode,
                                port=8050,
                                width='100%',
                                height=650)
