import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import plotly.graph_objects as go
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate

from irviz.graphs import SliceGraph, SpectraPlotGraph, PairPlotGraph


class Viewer(html.Div):
    _global_slicer_counter = 0

    def __init__(self, app, data, decomposition=None, bounds=None, ):
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
        self.pair_plot_graph = PairPlotGraph(self.decomposition, self)

        # Initialize configuration bits

        # Switches for views
        # Show Spectra, Show Decomposition, Show Pair Plot, Show Orthogonal Slices
        view_checklist = dbc.FormGroup(
            [
                dbc.Label("Toggle Views"),
                dbc.Checklist(
                    options=[
                        {"label": "Show Spectra", "value": "show_spectra"},
                        {"label": "Show Decomposition", "value": "show_decomposition"},
                        {"label": "Show Pair Plot", "value": "show_pair_plot"},
                        {"label": "Show Orthogonal Slices", "value": "show_orthogonal_slices"}
                    ],
                    value=["show_spectra", "show_decomposition", "show_pair_plot"],
                    id="view-checklist",
                    switch=True,
                )
            ]
        )
        view_selector = html.Div(
            [
                dbc.Form([view_checklist]),
            ]
        )

        radio_kwargs = dict(className='btn-group',
                            labelClassName="btn btn-secondary",
                            labelCheckedClassName="active",
                            options=[
                                {'label': 'Component 1', 'value': 0},
                                {'label': 'Component 2', 'value': 1},
                                {'label': 'Component 3', 'value': 2}
                            ])

        self.decomposition_component_selector = dbc.RadioItems(id='decomposition-component-selector', value=0,
                                                               **radio_kwargs)

        decomposition_selector_layout = html.Div(
            [
                html.P(id="decomposition-component-selector-p", className="card-text",
                       children="Decomposition Component"),
                self.decomposition_component_selector
            ],
            className='radio-group'
        )

        self.decomposition_component_1 = dbc.RadioItems(id='component-selector-1', value=0, **radio_kwargs)
        self.decomposition_component_2 = dbc.RadioItems(id='component-selector-2', value=1, **radio_kwargs)

        pair_plot_component_selector = html.Div(
            [
                html.P(id="pair-plot-component-selector-p", className="card-text", children="Pair Plot Components"),
                self.decomposition_component_1,
                html.Br(),
                self.decomposition_component_2,
            ],
            className='radio-group',
        )

        # Configuration layout
        config_children = html.Div(id="config-content",
                                   children=[
                                       view_selector,
                                       decomposition_selector_layout,
                                       pair_plot_component_selector
                                   ])

        config_view = dbc.Card(
            [
                dbc.CardHeader(
                    dbc.Tabs(
                        [
                            dbc.Tab(label="Settings", tab_id="settings-tab"),
                            dbc.Tab(label="Info", tab_id="info-tab"),
                        ],
                        id="config-tabs",
                        card=True,
                        active_tab="settings-tab",
                    )
                ),
                dbc.CardBody(config_children),
            ]
        )

        # Set up callbacks (Graphs need to wait until all children in this viewer are init'd)
        self.spectra_graph.register_callbacks()
        self.slice_graph.register_callbacks()
        self.pair_plot_graph.register_callbacks()

        # Initialize layout
        children = html.Div([config_view,
                             self.slice_graph,
                             self.spectra_graph,
                             self.decomposition_graph,
                             self.pair_plot_graph])

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

    div = html.Div(children=[viewer])  # , viewer2])
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
