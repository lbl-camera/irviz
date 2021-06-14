import dash_bootstrap_components as dbc
import dash_html_components as html
from dash_core_components import Graph

from irviz.graphs import DecompositionGraph, MapGraph, PairPlotGraph, SpectraPlotGraph


class Viewer(html.Div):
    _global_slicer_counter = 0

    def __init__(self, app, data, decomposition=None, bounds=None):
        self.data = data
        self._app = app
        self.decomposition = decomposition
        self.bounds = bounds

        Viewer._global_slicer_counter += 1

        # Initialize graphs
        spectra_graph_labels = {'xaxis_title': 'Wavenumber (cm⁻¹)'}
        self.spectra_graph = SpectraPlotGraph(data, bounds, self, labels=spectra_graph_labels)
        map_graph_labels = {'xaxis_title': 'X (μ)', 'yaxis_title': 'Y (μ)'}
        self.map_graph = MapGraph(data, bounds, self, labels=map_graph_labels)
        # self.orthogonal_x_graph = SliceGraph(data, self)
        # self.orthogonal_y_graph = SliceGraph(data, self)
        if self.decomposition is not None:
            decomposition_graph_labels = map_graph_labels
            self.decomposition_graph = DecompositionGraph(self.decomposition,
                                                          bounds,
                                                          self,
                                                          labels=decomposition_graph_labels)
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
        decomposition_selector_layout = html.Div()
        pair_plot_component_selector = html.Div()
        if self.decomposition is not None:
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
                    html.H3(id="decomposition-component-selector-p", className="card-text",
                           children="Decomposition Component"),
                    self.decomposition_component_selector
                ],
                className='radio-group'
            )

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

    @property
    def spectrum(self):
        """The currently shown spectrum energy/wavenumber and intensity values"""
        return self.spectra_graph.spectrum

    @property
    def spectral_value(self):
        """The current value of the crosshair position in energy/wavenumber"""
        return self.spectra_graph.spectral_value

    @property
    def spectral_index(self):
        """The current index of the crosshair position along the energy/wavenumber domain"""
        return self.spectra_graph.spectral_index

    @property
    def intensity(self):
        """The intensity value of the crosshair position"""
        return self.spectra_graph.intensity

    @property
    def position(self):
        """The spatial position of the current spectrum"""
        return self.spectra_graph.position

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
