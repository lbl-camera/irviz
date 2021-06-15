import dash_bootstrap_components as dbc
import dash_html_components as html
from dash_core_components import Graph, Slider

from irviz.components import ColorScaleSelector
from irviz.graphs import DecompositionGraph, MapGraph, PairPlotGraph, SpectraPlotGraph, decomposition_color_scales


class Viewer(html.Div):
    _global_slicer_counter = 0

    def __init__(self, app, data, decomposition=None, bounds=None):
        self.data = data
        self._app = app
        self.decomposition = decomposition
        self.bounds = bounds

        Viewer._global_slicer_counter += 1

        # Initialize graphs
        self.spectra_graph = SpectraPlotGraph(data, bounds, self)
        self.map_graph = MapGraph(data, bounds, self)
        # self.orthogonal_x_graph = SliceGraph(data, self)
        # self.orthogonal_y_graph = SliceGraph(data, self)
        if self.decomposition is not None:
            self.decomposition_graph = DecompositionGraph(self.decomposition, bounds, self)
            self.pair_plot_graph = PairPlotGraph(self.decomposition, self)
        else:
            self.decomposition_graph = Graph(id='empty-decomposition-graph', style={'display': 'none'})
            self.pair_plot_graph = Graph(id='empty-pair-plot-graph', style={'display': 'none'})

        # Initialize configuration bits
        # TODO: callback for switching tabs and rendering the Card content

        # Switches for views
        initial_views = ["show_spectra"]
        if self.decomposition is not None:
            initial_views.extend(["show_decomposition", "show_pair_plot"])
        self.graph_toggles = dbc.Checklist(
            options=[
                {"label": "Show Spectra", "value": "show_spectra"},
                {"label": "Show Decomposition", "value": "show_decomposition", "disabled": self.decomposition is None},
                {"label": "Show Pair Plot", "value": "show_pair_plot", "disabled": self.decomposition is None},
                {"label": "Show Orthogonal Slices", "value": "show_orthogonal_slices"}
            ],
            value=initial_views,
            id="view-checklist",
            switch=True,
        )
        view_switches = dbc.FormGroup(
            [
                html.H3("Toggle Views"),
                self.graph_toggles
            ]
        )
        view_selector = dbc.Form([view_switches])

        # Decomposition and pair plot component selectors
        pair_plot_component_selector = html.Div()
        if self.decomposition is not None:
            radio_kwargs = dict(className='btn-group-vertical col-sm-auto',
                                labelClassName="btn btn-secondary",
                                labelCheckedClassName="active",
                                options=[{'label': f'{i+1}', 'value': i}
                                         for i in range(decomposition.shape[0])]

                                )

            self.decomposition_component_selector = dbc.Checklist(id='decomposition-component-selector',
                                                                  value=[0],
                                                                  style={'paddingLeft':0, 'paddingRight':0},
                                                                  **radio_kwargs)

            self.component_opacity_sliders = html.Div(
                [Slider(
                    id={'type': 'component-opacity',
                        'index': i},
                    min=0,
                    max=1,
                    step=.1,
                    value=.5 if i else 1,
                    className='centered-slider'
                ) for i in range(self.decomposition.shape[0])],
                className='col-sm',
                style={'paddingLeft':0, 'paddingRight':0}
            )

            self.component_color_scale_selectors = html.Div(
                [ColorScaleSelector(app,
                                    {'type':'color-scale-selector',
                                    'index': i},
                                    values=decomposition_color_scales,
                                    value=decomposition_color_scales[i % len(decomposition_color_scales)]
                                    )
                 for i in range(self.decomposition.shape[0])],
                className='col-sm-auto',
                style={'paddingLeft':0, 'paddingRight':0, 'marginTop':2.5},
            )

            decomposition_selector_layout = html.Div(
                [
                    html.H3(id="decomposition-component-selector-p", className="card-text",
                            children="Decomposition Component"),
                    html.Div([
                        html.Div([self.decomposition_component_selector,
                                  self.component_color_scale_selectors,
                                  self.component_opacity_sliders,],
                                 className='row well'
                                 ),
                    ],
                    className='container'
                    ),
                ],
                className='radio-group'
            )

            radio_kwargs['className'] = 'btn-group'  # wipe out other classes

            self.decomposition_component_1 = dbc.RadioItems(id='component-selector-1', value=0, **radio_kwargs)
            self.decomposition_component_2 = dbc.RadioItems(id='component-selector-2', value=1, **radio_kwargs)

            pair_plot_component_selector = dbc.FormGroup(
                [
                    html.H3(id='pair-plot-component-selector-p', className='card-text', children="Pair Plot Components"),
                    self.decomposition_component_1,
                    html.Br(),
                    self.decomposition_component_2,
                ],
                className='radio-group',
            )

        # Settings tab layout
        # TODO put in function so we can use with callback
        settings_children = [view_selector]
        if decomposition is not None:
            settings_children.extend([decomposition_selector_layout, pair_plot_component_selector])
        settings_layout = dbc.Card(
            dbc.CardBody(children=settings_children)
        )

        # Info tab layout
        # TODO
        self.info_content = html.Div(id='info-content', children=["info"])
        info_layout = dbc.Card(dbc.CardBody(children=[self.info_content]))

        # Create the entire configuration layout
        config_view = dbc.Tabs(id='config-view',
                               children=[
                                   dbc.Tab(label="Settings", tab_id="settings-tab",
                                           children=settings_layout),
                                   dbc.Tab(label="Info", tab_id="info-tab", children=info_layout),
                               ],
                               )

        # Initialize layout
        layout_div_children = [self.map_graph,
                               self.decomposition_graph,
                               config_view,
                               self.spectra_graph,
                               self.pair_plot_graph]
        children = html.Div(children=layout_div_children,
                            className='row well')

        # Set up callbacks (Graphs need to wait until all children in this viewer are init'd)
        self.spectra_graph.register_callbacks()
        self.map_graph.register_callbacks()
        if self.decomposition is not None:
            self.pair_plot_graph.register_callbacks()
            self.decomposition_graph.register_callbacks()

        super(Viewer, self).__init__(children=children,
                                     className='container-fluid',
                                     )


def notebook_viewer(data, decomposition=None, bounds=None, mode='inline', width='100%', height=650):
    was_running = True
    from irviz.utils import dash as irdash
    try:
        from jupyter_dash import JupyterDash
    except ImportError:
        print("Please install jupyter-dash first.")
    else:
        if not irdash.app:
            # Creating a new app means we never ran the server
            irdash.app = JupyterDash(__name__)
            was_running = False

    viewer = Viewer(irdash.app, data.compute(), decomposition, bounds)
    # viewer2 = Viewer(data.compute(), app=app)

    div = html.Div(children=[viewer])  # , viewer2])
    irdash.app.layout = div

    # Prevent server from being run multiple times; only one instance allowed
    if not was_running:
        irdash.app.run_server(mode=mode)
    else:
        # Values passed here are from
        # jupyter_app.jupyter_dash.JupyterDash.run_server
        irdash.app._display_in_jupyter(dashboard_url='http://127.0.0.1:8050/',
                                mode=mode,
                                port=8050,
                                width=width,
                                height=height)
