import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import plotly.graph_objects as go
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
from irviz.graphs import SliceGraph


class Viewer(html.Div):
    _global_slicer_counter = 0

    def __init__(self, app, data, decomposition=None, bounds=None,):
        self.data = data
        self._app = app
        self.decomposition = decomposition
        self.bounds = bounds

        # Initialize graphs
        self.spectra_graph = dcc.Graph(id=f'spectra_{self._global_slicer_counter}')
        self.slice_graph = SliceGraph(data, self)
        # self.orthogonal_x_graph = SliceGraph(data, self)
        # self.orthogonal_y_graph = SliceGraph(data, self)
        self.decomposition_graph = SliceGraph(self.decomposition, self)
        self.pair_plot_graph = dcc.Graph(id=f'pair_plot_{self._global_slicer_counter}')

        # Initialize layout
        children = html.Div([self.slice_graph, self.spectra_graph, self.decomposition_graph])

        super(Viewer, self).__init__(children=children,
                                     style={'display': 'grid',
                                            'gridTemplateColumns': '50% 50%',
                                            },
                                     )

        # Bind callbacks
        self._app.callback(
            Output(f'spectra_{self._global_slicer_counter}', 'figure'),
            Input(self.slice_graph.id, 'clickData'))(self.update_spectra_plot_new)

        # Initialize views (TODO)

    def update_spectra_plot_new(self, click_data):
        # We need all the slicer state data ready; otherwise, don't update any Graph objects
        if click_data is None:
            raise PreventUpdate
        print("CLICK_DATA: ", click_data)

        y_index = click_data["points"][0]["y"]
        x_index = click_data["points"][0]["x"]

        y = self.data[:, y_index, x_index]
        x = np.arange(0, self.data.shape[0])
        fig = go.Figure(data={'type': 'scatter', 'x': x, 'y': y})
        fig.update_layout(title=f'Spectra Intensities @ (x: {x_index}, y: {y_index})',
                          xaxis_title="Spectra",
                          yaxis_title="Intensity")
        return fig


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
