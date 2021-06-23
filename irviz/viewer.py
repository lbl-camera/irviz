import numbers
import warnings

import dash_bootstrap_components as dbc
import dash_html_components as html
import numpy as np
from dash_core_components import Graph, Slider

from irviz.components import ColorScaleSelector
from irviz.graphs import DecompositionGraph, MapGraph, PairPlotGraph, SpectraPlotGraph, decomposition_color_scales


# TODO: organize Viewer.__init__ (e.g. make a validation method)

# TODO: update docstrings for annotation; validate annotation

class Viewer(html.Div):
    """Interactive viewer that creates and contains all of the visualized components within the Dash app"""

    _global_slicer_counter = 0

    def __init__(self,
                 app,
                 data,
                 decomposition=None,
                 bounds=None,
                 cluster_labels=None,
                 cluster_label_names=None,
                 component_spectra=None,
                 x_axis_title='X',
                 y_axis_title='Y',
                 spectra_axis_title='Spectral Units',
                 intensity_axis_title='Intensity',
                 invert_spectra_axis=False,
                 annotations=None):
        """Viewer on the Dash app.

        Provides some properties for accessing visualized data (e.g. the current spectrum).

        Parameters
        ----------
        app : dash.Dash or jupyter_dash.JupyterDash
            Reference to the Dash application to add components to
        data : dask.array
            3D array data (with axes E/wave-number, Y, and X)
        decomposition : np.ndarray
            (optional) Decomposition of the data
        bounds : list
            List of min, max pairs that define each axis's lower and upper bounds
        cluster_labels : np.ndarray
            Array that contains cluster integer labels over the Energy (wavenumber) axis
        cluster_label_names : list
            List of names for each label in the cluster label array
        component_spectra : list or np.ndarray
            List of component spectra of the decomposition
        x_axis_title : str
            Title of the x-axis for the rendered data and decomposition figures
        y_axis_title : str
            Title of the y-axis for the rendered data and decomposition figures
        spectra_axis_title : str
            Title for the spectra axis in the rendered spectra plot
        intensity_axis_title : str
            Title for the intensity axis in the rendered spectra plot
        invert_spectra_axis : bool
            Whether or not to invert the spectra axis on the spectra plot
        annotations : dict[str, dict]
            Dictionary that contains annotation names that map to annotations.
            The annotation dictionaries support the following keys:
                'range' : list or tuple of length 2
                'position' : number
                'color' : color (hex str, rgb str, hsl str, hsv str, named CSS color)
            Example:
                annotations={
                    'x': {
                        'range': (1000, 1500),
                        'color': 'green'
                    },
                    'y': {
                        'position': 300,
                        'range': [200, 500]
                    },
                    'z': {
                        'position': 900,
                        'color': '#34afdd'
                    }
                }
        """

        Viewer._global_slicer_counter += 1
        self.data = data
        self._app = app
        self.decomposition = decomposition

        self.bounds = np.asarray(bounds)
        if self.bounds.shape != (3, 2):  # bounds should contain a min/max pair for each dimension
            self.bounds = [[0, self._data.shape[0] - 1],
                           [0, self._data.shape[1] - 1],
                           [0, self._data.shape[2] - 1]]

        # Validate annotations TODO: reorganize
        if annotations is not None:
            for name, annotation in annotations.items():
                r = annotation.get('range', None)
                p = annotation.get('position', None)
                if r is not None and p is not None:
                    if not (r[0] <= p <= r[1]):
                        warnings.warn(f"position {p} is not within the range {r}")
                for kwarg, value in annotation.items():
                    # Range must be an iterable of length 2
                    if kwarg == "range":
                        try:
                            iter(value)
                        except TypeError:
                            raise TypeError(f"'range' must contain a tuple or list as its value")
                        else:
                            if len(value) != 2:
                                raise ValueError(f"'range' must contain a list/tuple of two numerical values")
                    # Position must be a number
                    elif kwarg == "position":
                        if not isinstance(value, numbers.Real):
                            raise ValueError(f"'position' must be a numerical value")
                    # Only color names supported right now
                    elif kwarg == "color":
                        if not isinstance(value, str):
                            raise TypeError(f"'color' must be a color name (string)")
                    else:
                        raise ValueError(f"'{kwarg}' is not currently supported as a keyword in annotations")

        # Component spectra shape should be (#components, #wavenumber)
        component_spectra_array = np.asarray(component_spectra)
        if len(component_spectra_array.shape) > 0:
            if (component_spectra_array.shape[0] != self.decomposition.shape[0]
                    or component_spectra_array.shape[1] != self.data.shape[0]):
                warnings.warn(f"The provided 'component_spectra' does not have a valid shape: "
                              f"{component_spectra_array.shape}; "
                              f"shape should be number of components, number of energies (wave-numbers).")

        # Initialize graphs
        self.spectra_graph = SpectraPlotGraph(data,
                                              self.bounds,
                                              self,
                                              component_spectra=component_spectra,
                                              xaxis_title=spectra_axis_title,
                                              yaxis_title=intensity_axis_title,
                                              invert_spectra_axis=invert_spectra_axis,
                                              annotations=annotations)
        self.map_graph = MapGraph(data, self.bounds, cluster_labels, cluster_label_names, self, xaxis_title=x_axis_title, yaxis_title=y_axis_title)
        # self.orthogonal_x_graph = SliceGraph(data, self)
        # self.orthogonal_y_graph = SliceGraph(data, self)
        if self.decomposition is not None:
            self.decomposition_graph = DecompositionGraph(self.decomposition,
                                                          self.bounds,
                                                          self,
                                                          xaxis_title=x_axis_title,
                                                          yaxis_title=y_axis_title)
            self.pair_plot_graph = PairPlotGraph(self.decomposition, cluster_labels, cluster_label_names, self)
        else:
            self.decomposition_graph = Graph(id='empty-decomposition-graph', style={'display': 'none'})
            self.pair_plot_graph = Graph(id='empty-pair-plot-graph', style={'display': 'none'})

        # Initialize configuration bits
        # TODO: callback for switching tabs and rendering the Card content

        # Switches for views
        initial_views = ["show_spectra"]
        if self.decomposition is not None:
            initial_views.extend(["show_decomposition", "show_pair_plot"])
        if cluster_labels is not None:
            initial_views.append('show_clusters')
        self.graph_toggles = dbc.Checklist(
            options=[
                {"label": "Show Spectra", "value": "show_spectra"},
                {"label": "Show Decomposition", "value": "show_decomposition", "disabled": self.decomposition is None},
                {"label": "Show Pair Plot", "value": "show_pair_plot", "disabled": self.decomposition is None},
                {"label": "Show Orthogonal Slices", "value": "show_orthogonal_slices"},
                {"label": "Show Cluster Labels", "value": "show_clusters", "disabled": cluster_labels is None}
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
                    className='centered-slider',
                    disabled=True if i else False
                ) for i in range(self.decomposition.shape[0])],
                className='col-sm',
                style={'paddingLeft':0, 'paddingRight':0},
                id='component-opacity-sliders'
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

        self.map_color_scale_selector = ColorScaleSelector(app=self._app, _id='map-color-scale-selector', value='Viridis')

        map_settings_form = dbc.Form(dbc.FormGroup([html.H3("Map Color Scale"), self.map_color_scale_selector]))

        # Views layout
        map_layout = dbc.Card(
            dbc.CardBody(children=[map_settings_form])
        )

        # Views layout
        views_layout = dbc.Card(
            dbc.CardBody(children=[view_selector])
        )

        # Settings tab layout
        # TODO put in function so we can use with callback
        decomposition_layout = dbc.Card(
            dbc.CardBody(children=[decomposition_selector_layout, pair_plot_component_selector])
        )

        # Info tab layout
        # TODO
        self.info_content = html.Div(id='info-content', children=["info"])
        info_layout = dbc.Card(dbc.CardBody(children=[self.info_content]))

        tabs = [dbc.Tab(label='Map', tab_id='map-tab', children=map_layout),
                dbc.Tab(label='Views', tab_id='views-tab', children=views_layout),
                dbc.Tab(label="Info", tab_id="info-tab", children=info_layout),
                ]
        if decomposition is not None:
            tabs.insert(1, dbc.Tab(label="Decomposition", tab_id="settings-tab", children=decomposition_layout))

        # Create the entire configuration layout
        config_view = html.Div(dbc.Tabs(id='config-view', children=tabs), className='col-lg-4')

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


def notebook_viewer(data,
                    decomposition=None,
                    bounds=None,
                    component_spectra=None,
                    spectra_axis_title='',
                    intensity_axis_title='',
                    x_axis_title='',
                    y_axis_title='',
                    invert_spectra_axis=False,
                    cluster_labels=None,
                    cluster_label_names=None,
                    mode='inline',
                    width='100%',
                    height=650,
                    annotations=None):
    """Create a Viewer inside of a Jupyter Notebook or Lab environment.

    Parameters
    ----------
    data : dask.array
        3D data to visualize in the Viewer
    decomposition : np.ndarray
        Component values for the decomposed data
    bounds : list
        List of min, max pairs that define each axis's lower and upper bounds
    cluster_labels : np.ndarray
            Array that contains cluster integer labels over the Energy (wavenumber) axis
    cluster_label_names : list
        List of names for each label in the cluster label array
    component_spectra : list or np.ndarray
        List of component spectra of the decomposition
    x_axis_title : str
        Title of the x-axis for the rendered data and decomposition figures
    y_axis_title : str
        Title of the y-axis for the rendered data and decomposition figures
    spectra_axis_title : str
        Title for the spectra axis in the rendered spectra plot
    intensity_axis_title : str
        Title for the intensity axis in the rendered spectra plot
    invert_spectra_axis : bool
        Whether or not to invert the spectra axis on the spectra plot
    mode : str
        Defines where the Viewer app is displayed (default is 'inline')
    width : int or str
        CSS-style width value that defines the width of the rendered Viewer app
    height : int or str
        CSS-style height value that defines the height of the rendered Viewer app
    annotations : dict[str, dict]
        Dictionary that contains annotation names that map to annotations.
        The annotation dictionaries support the following keys:
            'range' : list or tuple of length 2
            'position' : number
            'color' : color (hex str, rgb str, hsl str, hsv str, named CSS color)
        Example:
            annotations={
                'x': {
                    'range': (1000, 1500),
                    'color': 'green'
                },
                'y': {
                    'position': 300,
                    'range': [200, 500]
                },
                'z': {
                    'position': 900,
                    'color': '#34afdd'
                }
            }

    Returns
    -------
    viewer
        Returns a reference to the created Viewer, which acts as a handle to the Dash app.
        This is useful for accessing data inside of the Viewer (via its properties).

    """
    was_running = True
    app_kwargs = {'external_stylesheets': [dbc.themes.BOOTSTRAP]}
    from irviz.utils import dash as irdash
    try:
        from jupyter_dash import JupyterDash
    except ImportError:
        print("Please install jupyter-dash first.")
    else:
        if not irdash.app:
            # Creating a new app means we never ran the server
            irdash.app = JupyterDash(__name__, **app_kwargs)
            was_running = False

    viewer = Viewer(irdash.app,
                    data.compute(),
                    decomposition=decomposition,
                    component_spectra=component_spectra,
                    bounds=bounds,
                    x_axis_title=x_axis_title,
                    y_axis_title=y_axis_title,
                    cluster_labels=cluster_labels,
                    cluster_label_names=cluster_label_names,
                    spectra_axis_title=spectra_axis_title,
                    intensity_axis_title=intensity_axis_title,
                    invert_spectra_axis=invert_spectra_axis,
                    annotations=annotations)
    # viewer2 = Viewer(data.compute(), app=app)

    div = html.Div(children=[viewer])  # , viewer2])
    irdash.app.layout = div

    # Prevent server from being run multiple times; only one instance allowed
    if not was_running:
        irdash.app.run_server(mode=mode,
                              width=width,
                              height=height)
    else:
        # Values passed here are from
        # jupyter_app.jupyter_dash.JupyterDash.run_server

        if irdash.app._in_colab:
            irdash.app._display_in_colab(dashboard_url='http://127.0.0.1:8050/',
                                         mode=mode,
                                         port=8050,
                                         width=width,
                                         height=height)
        else:
            irdash.app._display_in_jupyter(dashboard_url='http://127.0.0.1:8050/',
                                           mode=mode,
                                           port=8050,
                                           width=width,
                                           height=height)

    return viewer
