import numbers
import warnings
from itertools import count
from typing import Callable, Any

import dash
import dash_bootstrap_components as dbc
import dash_html_components as html
import numpy as np
from dash_core_components import Graph, Slider
from dash.dependencies import Input, Output, State
from nptyping import NDArray

from irviz.components import spectra_annotation_dialog
from irviz.components.color_scale_selector import ColorScaleSelector
from irviz.components.modal_dialogs import slice_annotation_dialog
from irviz.graphs import DecompositionGraph, MapGraph, OpticalGraph, PairPlotGraph, SpectraPlotGraph
from irviz.graphs._colors import decomposition_color_scales
from irviz.utils.dash import targeted_callback
from irviz.utils.math import nearest_bin
from irviz.utils.strings import phonetic_from_int


# TODO: organize Viewer.__init__ (e.g. make a validation method)
# TODO: update docstrings for annotation; validate annotation
# TODO: JSON schema validation for annotations
# TODO: functools.wraps for notebook_viewer


class Viewer(html.Div):
    """Interactive viewer that creates and contains all of the visualized components within the Dash app"""

    _instance_counter = count(0)

    def __init__(self,
                 app,
                 data,
                 optical=None,
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
                 annotations=None,
                 error_func=None):
        """Viewer on the Dash app.

        Provides some properties for accessing visualized data (e.g. the current spectrum).

        Parameters
        ----------
        app : dash.Dash or jupyter_dash.JupyterDash
            Reference to the Dash application to add components to
        data : dask.array
            3D array data (with axes E/wave-number, Y, and X)
        optical : np.ndarray
            (optional) An optical image registered with the spectral data
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
        annotations : List[dict]
            Dictionary that contains annotation names that map to annotations.
            The annotation dictionaries support the following keys:
                'name' : the name of the annotation
                'range' : list or tuple of length 2
                'position' : number
                'color' : color (hex str, rgb str, hsl str, hsv str, named CSS color)
            Example:
                annotations=[
                    {
                        'name': 'x',
                        'range': (1000, 1500),
                        'color': 'green'
                    },
                    {
                        'name': 'y',
                        'position': 300,
                        'range': [200, 500]
                    },
                    {   'name': 'z',
                        'position': 900,
                        'color': '#34afdd'
                    }
                ]
        error_func : Callable[[NDArray[(Any, Any)]], np.ndarray[Any]]
            A callable function that takes an array of shape (E, N), where E is the length of the spectral dimension and
            N is the number of curves over which to calculate error. The return value is expected to be a 1-D array of
            length E. The default is to apply a std dev over the N axis.
        """
        self._app = app
        self._instance_index = next(self._instance_counter)

        # Normalize bounds
        if bounds is None or np.asarray(bounds).shape != (
        3, 2):  # bounds should contain a min/max pair for each dimension
            bounds = [[0, self._data.shape[0] - 1],
                      [0, self._data.shape[1] - 1],
                      [0, self._data.shape[2] - 1]]
        bounds = np.asarray(bounds)

        # Normalize labels
        if cluster_labels is not None and cluster_label_names is None:
            cluster_label_names = [phonetic_from_int(i) for i in range(np.unique(cluster_labels).size)]

        self._validate(data,
                       optical,
                       decomposition,
                       bounds,
                       cluster_labels,
                       cluster_label_names,
                       component_spectra,
                       x_axis_title,
                       y_axis_title,
                       spectra_axis_title,
                       intensity_axis_title,
                       invert_spectra_axis,
                       annotations,
                       error_func)

        # Initialize graphs
        self.spectra_graph = SpectraPlotGraph(data,
                                              bounds,
                                              self,
                                              decomposition=decomposition,
                                              component_spectra=component_spectra,
                                              xaxis_title=spectra_axis_title,
                                              yaxis_title=intensity_axis_title,
                                              invert_spectra_axis=invert_spectra_axis,
                                              annotations=annotations,
                                              error_func=error_func)
        self.map_graph = MapGraph(data, bounds, cluster_labels, cluster_label_names, self, xaxis_title=x_axis_title, yaxis_title=y_axis_title)
        if optical is not None:
            self.optical_graph = OpticalGraph(data, optical, bounds, cluster_labels, cluster_label_names, self, xaxis_title=x_axis_title, yaxis_title=y_axis_title)
        else:
            self.optical_graph = Graph(id='empty-optical-graph', style={'display': 'none'})
        # self.orthogonal_x_graph = SliceGraph(data, self)
        # self.orthogonal_y_graph = SliceGraph(data, self)
        if decomposition is not None:
            self.decomposition_graph = DecompositionGraph(decomposition,
                                                          bounds,
                                                          cluster_labels,
                                                          cluster_label_names,
                                                          self,
                                                          xaxis_title=x_axis_title,
                                                          yaxis_title=y_axis_title)
            self.pair_plot_graph = PairPlotGraph(decomposition, bounds, cluster_labels, cluster_label_names, self)
        else:
            self.decomposition_graph = Graph(id='empty-decomposition-graph', style={'display': 'none'})
            self.pair_plot_graph = Graph(id='empty-pair-plot-graph', style={'display': 'none'})


        # Initialize configuration bits

        # Switches for views
        initial_views = ["show_spectra"]
        if decomposition is not None:
            initial_views.extend(["show_decomposition", "show_pair_plot"])
        if optical is not None:
            initial_views.append('show_optical')
        if cluster_labels is not None:
            initial_views.append('show_clusters')
        self._graph_toggles = dbc.Checklist(
            options=[
                {"label": "Show Spectra", "value": "show_spectra"},
                {"label": "Show Optical", 'value': 'show_optical', 'disabled': optical is None},
                {"label": "Show Decomposition", "value": "show_decomposition", "disabled": decomposition is None},
                {"label": "Show Pair Plot", "value": "show_pair_plot", "disabled": decomposition is None},
                {"label": "Show Orthogonal Slices", "value": "show_orthogonal_slices"},
                {"label": "Show Cluster Labels", "value": "show_clusters", "disabled": cluster_labels is None}
            ],
            value=initial_views,
            id="view-checklist",
            switch=True,
        )
        view_switches = dbc.FormGroup(
            [
                dbc.Label("Toggle Views"),
                self._graph_toggles
            ]
        )
        view_selector = dbc.Form([view_switches])

        # Decomposition and pair plot component selectors
        decomposition_selector_layout = html.Div()
        pair_plot_component_selector = html.Div()
        if decomposition is not None:
            radio_kwargs = dict(className='btn-group-vertical col-sm-auto',
                                labelClassName="btn btn-secondary",
                                labelCheckedClassName="active",
                                options=[{'label': f'{i+1}', 'value': i}
                                         for i in range(decomposition.shape[0])]

                                )

            self._decomposition_component_selector = dbc.Checklist(id='decomposition-component-selector',
                                                                   value=[0],
                                                                   style={'paddingLeft': 0, 'paddingRight': 0},
                                                                   **radio_kwargs)

            self._component_opacity_sliders = html.Div(
                [Slider(
                    id={'type': 'component-opacity',
                        'index': i},
                    min=0,
                    max=1,
                    step=.1,
                    value=.5 if i else 1,
                    className='centered-slider',
                    disabled=True if i else False
                ) for i in range(decomposition.shape[0])],
                className='col-sm',
                style={'paddingLeft': 0, 'paddingRight': 0},
                id='component-opacity-sliders'
            )

            self._component_color_scale_selectors = html.Div(
                [ColorScaleSelector(app,
                                    {'type':'color-scale-selector',
                                    'index': i},
                                    values=decomposition_color_scales,
                                    value=decomposition_color_scales[i % len(decomposition_color_scales)]
                                    )
                 for i in range(decomposition.shape[0])],
                className='col-sm-auto',
                style={'paddingLeft':0, 'paddingRight':0, 'marginTop':2.5},
            )

            decomposition_selector_layout = dbc.FormGroup(
                [
                    dbc.Label(id="decomposition-component-selector-p", className="card-text",
                            children="Decomposition Component"),
                    html.Div([
                        html.Div([self._decomposition_component_selector,
                                  self._component_color_scale_selectors,
                                  self._component_opacity_sliders, ],
                                 className='row well'
                                 ),
                    ],
                    className='container'
                    ),
                ],
                className='radio-group'
            )

            radio_kwargs['className'] = 'btn-group'  # wipe out other classes

            self._decomposition_component_1 = dbc.RadioItems(id='component-selector-1', value=0, **radio_kwargs)
            radio_kwargs = radio_kwargs.copy()
            radio_kwargs['options'] = radio_kwargs['options'].copy() + [{'label': 'ALL', 'value': 'ALL'}]
            self._decomposition_component_2 = dbc.RadioItems(id='component-selector-2', value=1, **radio_kwargs)

            pair_plot_component_selector = dbc.FormGroup(
                [
                    dbc.Label(id='pair-plot-component-selector-p', className='card-text', children="Pair Plot Components"),
                    html.Br(),
                    self._decomposition_component_1,
                    html.Br(),
                    self._decomposition_component_2,
                ],
                className='radio-group',
            )

        self._map_color_scale_selector = ColorScaleSelector(app=self._app, _id='map-color-scale-selector', value='Viridis')
        self._cluster_overlay_opacity = Slider(id={'type': 'cluster-opacity'},
                                               min=0,
                                               max=1,
                                               step=.05,
                                               value=.3,
                                               className='centered-slider',
                                               disabled=True if cluster_labels is None else False,
                                               )

        map_settings_form = dbc.Form([dbc.FormGroup([dbc.Label("Map Color Scale"), self._map_color_scale_selector]),
                                     dbc.FormGroup([dbc.Label("Cluster Label Overlay Opacity"), self._cluster_overlay_opacity])])

        # Views layout
        map_layout = dbc.Card(
            dbc.CardBody(children=[map_settings_form])
        )

        # Views layout
        views_layout = dbc.Card(
            dbc.CardBody(children=[view_selector])
        )

        # Annotations layout
        self.spectra_graph_annotations = dbc.ListGroup(id='spectra-graph-annotations',
                                                       children=[])
        self.slice_graph_annotations = dbc.ListGroup(id='slice-graph-annotations',
                                                     children=[])
        self.spectra_graph_add_annotation = dbc.Button("Add Annotation", id='spectra-graph-add-annotation', n_clicks=0)
        self.slice_graph_add_annotation = dbc.Button("Annotate Selection", id='slice-graph-add-annotation', n_clicks=0)
        annotations_layout = dbc.Card(dbc.CardBody(children=[self.slice_graph_annotations,
                                                             self.slice_graph_add_annotation,
                                                             self.spectra_graph_annotations,
                                                             self.spectra_graph_add_annotation,
                                                             ]))

        # Settings tab layout
        # TODO put in function so we can use with callback
        decomposition_layout = dbc.Card(
            dbc.CardBody(children=[decomposition_selector_layout, pair_plot_component_selector])
        )

        # Info tab layout
        # TODO
        self._info_content = html.Div(id='info-content', children=["info"])
        info_layout = dbc.Card(dbc.CardBody(children=[self._info_content]))

        tabs = [dbc.Tab(label='Map', tab_id='map-tab', children=map_layout),
                dbc.Tab(label='Views', tab_id='views-tab', children=views_layout),
                dbc.Tab(label='Annotations', tab_id='annotations-tab', children=annotations_layout),
                dbc.Tab(label="Info", tab_id="info-tab", children=info_layout),
                ]
        if decomposition is not None:
            tabs.insert(1, dbc.Tab(label="Decomposition", tab_id="settings-tab", children=decomposition_layout))

        # Create the entire configuration layout
        config_view = html.Div(dbc.Tabs(id='config-view', children=tabs), className='col-lg-3')

        # Create the Toast (notification thingy)
        self._notifier = dbc.Toast("placeholder",
                                   id="notifier",
                                   header="Tip",
                                   is_open=False,
                                   dismissable=True,
                                   icon="info",
                                   duration=4000,
                                   # top: 66 positions the toast below the navbar
                                   style={"position": "fixed", "top": 66, "right": 10, "width": 350}
                                   )

        # Wireup toast chained callback
        targeted_callback(lambda _: True,
                          Input(self._notifier.id, 'children'),
                          Output(self._notifier.id, 'is_open'),
                          app=self._app)

        self.spectra_annotation_dialog = spectra_annotation_dialog(app,
                                                                   'spectra-annotation-dialog',
                                                                   success_callback=self._add_annotation_from_dialog,
                                                                   success_output=Output(self.spectra_graph_annotations.id, 'children'),
                                                                   open_input=Input(self.spectra_graph_add_annotation.id, 'n_clicks'))

        self.slice_annotation_dialog = slice_annotation_dialog(app,
                                                               'slice-annotation-dialog',
                                                               success_callback=self._add_slice_annotation_from_dialog,
                                                               success_output=Output(self.slice_graph_annotations.id, 'children'),
                                                               open_input=Input(self.slice_graph_add_annotation.id, 'n_clicks'))

        for annotation in annotations or []:
            self._add_spectra_annotation(annotation)

        # Initialize layout
        layout_div_children = [self.map_graph,
                               self.optical_graph,
                               self.decomposition_graph,
                               config_view,
                               self.spectra_graph,
                               self.pair_plot_graph,
                               self._notifier,
                               self.spectra_annotation_dialog,
                               self.slice_annotation_dialog]
        children = html.Div(children=layout_div_children,
                            className='row well')

        # Set up callbacks (Graphs need to wait until all children in this viewer are init'd)
        self.spectra_graph.register_callbacks()
        self.map_graph.register_callbacks()
        if optical is not None:
            self.optical_graph.register_callbacks()
        if decomposition is not None:
            self.pair_plot_graph.register_callbacks()
            self.decomposition_graph.register_callbacks()

        super(Viewer, self).__init__(children=children,
                                     className='container-fluid',
                                     )

    @property
    def map(self):
        """The currently displayed map slice at the current spectral index"""
        return self.map_graph.map

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
        return self.map_graph.position

    @property
    def position_index(self):
        """The spatial position of the current spectrum as an index (y, x)"""
        return self.map_graph.position_index

    @property
    def selection(self):
        """A mask array representing the current spatial selection region"""
        return self.map_graph.selection

    @property
    def selection_indices(self):
        """The indices of all currently selected points, returned as (y, x)"""
        return self.map_graph.selection_indices

    @property
    def annotations(self):
        # TODO: Add map annotations
        """User-defined annotations on the spectra graph"""
        return self.spectra_graph.annotations

    @property
    def slice_annotations(self):
        if not hasattr(self, 'map_graph'):
            return []
        return self.map_graph.annotations

    def _add_spectra_annotation(self, annotation):
        self.spectra_graph.add_annotation(annotation)
        self.spectra_graph_annotations.children += str(annotation)

    def _add_slice_annotation(self, annotation):
        for graph in [self.map_graph, self.optical_graph, self.decomposition_graph, self.spectra_graph]:
            if hasattr(graph, 'add_annotation'):
                graph.add_annotation(annotation)
        self.slice_graph_annotations.children += str(annotation)

    def _add_annotation_from_dialog(self, n_clicks):
        # Get the form input values
        input_states = dash.callback_context.states
        annotation = dict()
        if input_states['spectra-annotation-dialog-upper-bound.value'] is None:
            annotation['position'] = input_states['spectra-annotation-dialog-lower-bound.value']
        else:
            annotation['range'] = (input_states['spectra-annotation-dialog-lower-bound.value'], input_states['spectra-annotation-dialog-upper-bound.value'])

        annotation['name'] = input_states['spectra-annotation-dialog-name.value']
        
        # Color will come back as 'rgb': {'r': r, 'g': g, 'b': b, 'a': a},
        #     need to convert to plotly color: 'rgba(r,g,b,a)'
        color_picker = input_states['spectra-annotation-dialog-color-picker.value']
        rgb = color_picker['rgb']
        color = f"rgb({rgb['r']}, {rgb['g']}, {rgb['b']})"
        annotation['opacity'] = rgb['a']
        annotation['color'] = color
        annotation['type'] = 'spectrum'
        self._add_spectra_annotation(annotation)

        return self.spectra_graph_annotations.children

    def _add_slice_annotation_from_dialog(self, n_clicks):
        # Get the form input values
        input_states = dash.callback_context.states
        annotation = dict()
        annotation['name'] = input_states['slice-annotation-dialog-name.value']
        annotation['mask'] = self.map_graph._selection_mask.z
        annotation['type'] = 'slice'

        # Color will come back as 'rgb': {'r': r, 'g': g, 'b': b, 'a': a},
        #     need to convert to plotly color: 'rgba(r,g,b,a)'
        color_picker = input_states['slice-annotation-dialog-color-picker.value']
        rgb = color_picker['rgb']
        color = f"rgb({rgb['r']}, {rgb['g']}, {rgb['b']})"
        annotation['color'] = color
        annotation['opacity'] = rgb['a']

        self._add_slice_annotation(annotation)

        return self.slice_graph_annotations.children

    def _validate(self,
                  data,
                  optical,
                  decomposition,
                  bounds,
                  cluster_labels,
                  cluster_label_names,
                  component_spectra,
                  x_axis_title,
                  y_axis_title,
                  spectra_axis_title,
                  intensity_axis_title,
                  invert_spectra_axis,
                  annotations,
                  error_func):

        # Validate annotations TODO: reorganize
        if annotations is not None:
            for i in reversed(range(len(annotations))):
                annotation = annotations[i]
                r = annotation.get('range', None)
                p = annotation.get('position', None)
                if r is not None and p is not None:
                    # Cannot supply both position and range in same annotation, ignore position
                    warnings.warn(f"cannot supply both 'range' and 'position' in the same annotation; "
                                  f"ignoring 'position'")
                    annotation.pop('position')
                    # if not (r[0] <= p <= r[1]):
                    #     warnings.warn(f"position {p} is not within the range {r}")

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
                    elif kwarg != "name":
                        raise ValueError(f"'{kwarg}' is not currently supported as a keyword in annotations")

        # Component spectra shape should be (#components, #wavenumber)
        component_spectra_array = np.asarray(component_spectra)
        if len(component_spectra_array.shape) > 0:
            if (component_spectra_array.shape[0] != decomposition.shape[0]
                    or component_spectra_array.shape[1] != data.shape[0]):
                warnings.warn(f"The provided 'component_spectra' does not have a valid shape: "
                              f"{component_spectra_array.shape}; "
                              f"shape should be number of components, number of energies (wave-numbers).")


def notebook_viewer(data,
                    optical=None,
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
                    error_func=None,
                    mode='inline',
                    width='100%',
                    height=650,
                    annotations=None):
    """Create a Viewer inside of a Jupyter Notebook or Lab environment.

    Parameters
    ----------
    data : dask.array
        3D data to visualize in the Viewer
    optical : np.ndarray
        (optional) An optical image registered with the spectral data
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
    annotations : List[dict]
            Dictionary that contains annotation names that map to annotations.
            The annotation dictionaries support the following keys:
                'name' : the name of the annotation
                'range' : list or tuple of length 2
                'position' : number
                'color' : color (hex str, rgb str, hsl str, hsv str, named CSS color)
            Example:
                annotations=[
                    {
                        'name': 'x',
                        'range': (1000, 1500),
                        'color': 'green'
                    },
                    {
                        'name': 'y',
                        'position': 300,
                        'range': [200, 500]
                    },
                    {   'name': 'z',
                        'position': 900,
                        'color': '#34afdd'
                    }
                ]
    error_func : Callable[[NDArray[(Any, Any)]], np.ndarray[Any]]
            A callable function that takes an array of shape (E, N), where E is the length of the spectral dimension and
            N is the number of curves over which to calculate error. The return value is expected to be a 1-D array of
            length E. The default is to apply a std dev over the N axis.

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
                    data,
                    optical=optical,
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
                    annotations=annotations,
                    error_func=error_func)
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
