import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import plotly.graph_objects as go
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
from dash_slicer import VolumeSlicer

from irviz.graphs import SliceGraph


class Viewer(html.Div):
    _global_slicer_counter = 0

    def __init__(self, data, app, ):
        self.data = data
        self._app = app

        Viewer._global_slicer_counter += 1

        # z_slicer = VolumeSlicer(self._app, self.data, axis=0)
        # y_slicer = VolumeSlicer(self._app, self.data, axis=1)
        # x_slicer = VolumeSlicer(self._app, self.data, axis=2)
        #
        # # Override some of the layout changes that VolumeSlicer does
        # common_opts = {'xaxis_showticklabels': True,
        #                'yaxis_showticklabels': True,
        #                'xaxis_showgrid': True,
        #                'yaxis_showgrid': True,
        #                'margin': None}
        # z_slicer.graph.figure.update_layout(title="y(x)",
        #                                     xaxis_title="X",
        #                                     yaxis_title="Y",
        #                                     # xaxis_tick0=0,
        #                                     # xaxis_dtick=1,
        #                                     # yaxis_tick0=0,
        #                                     # xaxis_dtick=1,
        #                                     **common_opts)  # {"l": 0, "r": 0, "b": 30, "t": 0, "pad": 10})
        # y_slicer.graph.figure.update_layout(title="E(y)",
        #                                     # xaxis_range=[0, volume.shape[2]],
        #                                     xaxis_scaleanchor="y",
        #                                     xaxis_scaleratio=(self.data.shape[0] / self.data.shape[2]),
        #                                     xaxis_title="X",
        #                                     yaxis_title="Z",
        #                                     **common_opts
        #                                     )
        # x_slicer.graph.figure.update_layout(title="E(x)",
        #                                     xaxis_scaleanchor="y",
        #                                     xaxis_scaleratio=(self.data.shape[0] / self.data.shape[1]),
        #                                     xaxis_title="Y",
        #                                     yaxis_title="Z",
        #                                     **common_opts)
        #
        # # Describe our html elements for each slicer
        # z_view = html.Div([
        #     z_slicer.graph,
        #     z_slicer.slider,
        #     *z_slicer.stores
        # ])
        # y_view = html.Div([
        #     y_slicer.graph,
        #     y_slicer.slider,
        #     *y_slicer.stores
        # ])
        # x_view = html.Div([
        #     x_slicer.graph,
        #     x_slicer.slider,
        #     *x_slicer.stores
        # ])
        self.spectra_graph = dcc.Graph(id=f'spectra_{self._global_slicer_counter}')
        # Describe html layout for spectral plot
        spectra_plot = html.Div([
            self.spectra_graph,
        ])

        self.slice_graph = SliceGraph(data, self)
        slice_div = html.Div([self.slice_graph, spectra_plot])
        children = slice_div

        # children = [z_view,
        #             y_view,
        #             x_view,
        #             spectra_plot]

        super(Viewer, self).__init__(children=children,
                                     style={'display': 'grid',
                                            'gridTemplateColumns': '50% 50%',
                                            },
                                     )

        self._app.callback(
            Output(f'spectra_{self._global_slicer_counter}', 'figure'),
            Input(self.slice_graph.id, 'clickData'))(self.update_spectra_plot_new)

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


def notebook_viewer(data, decomposition=None, mode='inline'):
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
    viewer = Viewer(data.compute(), app=app)
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
