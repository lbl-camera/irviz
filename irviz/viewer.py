import json
import numbers
import warnings
from functools import cached_property
from itertools import count

import dash
import dash_bootstrap_components as dbc
import dash_html_components as html
import numpy as np
from dash.dependencies import ALL, Input, Output

import ryujin.utils.dash
from irviz.components import SpectraAnnotationDialog, SliceAnnotationDialog
from irviz.graphs import DecompositionGraph, MapGraph, OpticalGraph, PairPlotGraph, SpectraPlotGraph
from ryujin import ComposableDisplay
from ryujin.components import Panel
from ryujin.utils.dash import targeted_callback
from ryujin.utils.strings import phonetic_from_int

__all__ = ['Viewer']


# TODO: organize Viewer.__init__ (e.g. make a validation method)
# TODO: update docstrings for annotation; validate annotation
# TODO: JSON schema validation for annotations
# TODO: functools.wraps for notebook_viewer

class AnnotationsPanel(Panel):
    def __init__(self, instance_index):
        self._instance_index = instance_index
        self.spectra_graph_annotations = dbc.Nav(id={'type': 'spectra_annotations',
                                                     'index': instance_index,
                                                     'wildcard': True},
                                                 pills=True,
                                                 vertical='md',
                                                 children=[])
        self.slice_graph_annotations = dbc.Nav(id={'type': 'slice_annotations',
                                                   'index': instance_index,
                                                   'wildcard': True},
                                               pills=True,
                                               vertical='md',
                                               children=[])
        self.spectra_graph_add_annotation = dbc.Button("Annotate Spectra", id='spectra-graph-add-annotation', n_clicks=0)
        self.slice_graph_add_annotation = dbc.Button("Annotate Selection", id='slice-graph-add-annotation', n_clicks=0)

        annotations_layout_children = html.Div(className="row", children=[
            html.Div(className="col-6", children=[
                self.slice_graph_annotations,
                self.slice_graph_add_annotation
            ]),
            html.Div(className="col-6", children=[
                self.spectra_graph_annotations,
                self.spectra_graph_add_annotation
            ])
        ])

        super(AnnotationsPanel, self).__init__('Annotations', annotations_layout_children)


class Viewer(ComposableDisplay):
    """Interactive viewer that creates and contains all of the visualized components within the Dash app"""

    _instance_counter = count(0)
    _annotation_counter = count(0)

    def __init__(self,
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
        # Normalize bounds
        if bounds is None or np.asarray(bounds).shape != (
                3, 2):  # bounds should contain a min/max pair for each dimension
            bounds = [[0, data.shape[0] - 1],
                      [0, data.shape[1] - 1],
                      [0, data.shape[2] - 1]]
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

        self._data = data
        self._bounds = bounds
        self._cluster_labels = cluster_labels
        self._cluster_label_names = cluster_label_names
        self._optical = optical
        self._decomposition = decomposition
        self._component_spectra = component_spectra
        self._x_axis_title = x_axis_title
        self._y_axis_title = y_axis_title
        self._spectra_axis_title = spectra_axis_title
        self._intensity_axis_title = intensity_axis_title
        self._invert_spectra_axis = invert_spectra_axis
        self._annotations = annotations
        self._error_func = error_func

        super(Viewer, self).__init__()

    def init_components(self, *args, **kwargs):
        self.graphs = {}
        # Initialize graphs
        self.graphs['map_graph'] = MapGraph(self._data,
                                            self._instance_index,
                                            self._cluster_labels,
                                            self._cluster_label_names,
                                            bounds=self._bounds,
                                            xaxis_title=self._x_axis_title,
                                            yaxis_title=self._y_axis_title,
                                            graph_kwargs=dict(style=dict(display='flex',
                                                                         flexDirection='row',
                                                                         height='100%',
                                                                         minHeight='450px'),
                                                              className='col-lg-4 p-0',
                                                              responsive=True,
                                                              )
                                            )

        if self._optical is not None:
            self.graphs['optical_graph'] = OpticalGraph(self._data,
                                                        self._instance_index,
                                                        self._optical,
                                                        self._bounds,
                                                        self._cluster_labels,
                                                        self._cluster_label_names,
                                                        self,
                                                        xaxis_title=self._x_axis_title,
                                                        yaxis_title=self._y_axis_title,
                                                        graph_kwargs=dict(style=dict(display='flex',
                                                                                     flexDirection='row',
                                                                                     height='100%',
                                                                                     minHeight='450px'),
                                                                          className='col-lg-4 p-0',
                                                                          responsive=True,
                                                                          )
                                                        )
        # self.orthogonal_x_graph = SliceGraph(data, self)
        # self.orthogonal_y_graph = SliceGraph(data, self)

        if self._decomposition is not None:
            self.graphs['decomposition_graph'] = DecompositionGraph(self._decomposition,
                                                                    self._instance_index,
                                                                    self._cluster_labels,
                                                                    self._cluster_label_names,
                                                                    self._bounds,
                                                                    xaxis_title=self._x_axis_title,
                                                                    yaxis_title=self._y_axis_title,
                                                                    graph_kwargs=dict(style=dict(display='flex',
                                                                                                 flexDirection='row',
                                                                                                 height='100%',
                                                                                                 minHeight='450px'),
                                                                                      className='col-lg-4 p-0',
                                                                                      responsive=True,
                                                                                      )
                                                                    )

        self.graphs['spectra_graph'] = SpectraPlotGraph(self._data,
                                                        self._instance_index,
                                                        self._bounds,
                                                        decomposition=self._decomposition,
                                                        component_spectra=self._component_spectra,
                                                        xaxis_title=self._spectra_axis_title,
                                                        yaxis_title=self._intensity_axis_title,
                                                        invert_spectra_axis=self._invert_spectra_axis,
                                                        annotations=self._annotations,
                                                        error_func=self._error_func,
                                                        graph_kwargs=dict(style=dict(display='flex',
                                                                                     flexDirection='row',
                                                                                     height='100%',
                                                                                     minHeight='450px'),
                                                                          className='col-lg-9 p-0',
                                                                          responsive=True
                                                                          ))
        if self._decomposition is not None:
            self.graphs['pair_plot_graph'] = PairPlotGraph(self._instance_index,
                                                           self._decomposition,
                                                           self._bounds,
                                                           self._cluster_labels,
                                                           self._cluster_label_names,
                                                           graph_kwargs=dict(className='col-lg-3 p-0',
                                                                             responsive=True,
                                                                             style=dict(display='flex',
                                                                                        flexDirection='row',
                                                                                        height='100%',
                                                                                        minHeight='450px')))

        self.annotations_panel = AnnotationsPanel(self._instance_index)

        self.spectra_annotation_dialog = SpectraAnnotationDialog('spectra-annotation-dialog',
                                                                   success_callback=self._add_annotation_from_dialog,
                                                                   success_output=Output(
                                                                       self.annotations_panel.spectra_graph_annotations.id,
                                                                       'children'),
                                                                   open_input=Input(
                                                                       self.annotations_panel.spectra_graph_add_annotation.id,
                                                                       'n_clicks'))

        self.slice_annotation_dialog = SliceAnnotationDialog('slice-annotation-dialog',
                                                               success_callback=self._add_slice_annotation_from_dialog,
                                                               success_output=Output(
                                                                   self.annotations_panel.slice_graph_annotations.id,
                                                                   'children'),
                                                               open_input=Input(
                                                                   self.annotations_panel.slice_graph_add_annotation.id,
                                                                   'n_clicks'))

        for annotation in self._annotations or []:
            self._add_spectra_annotation(annotation)

        return list(self.graphs.values()) + [self.spectra_annotation_dialog,
                                             self.slice_annotation_dialog]

    def init_callbacks(self, app):
        super(Viewer, self).init_callbacks(app)

        # spectra annotation removal callback
        targeted_callback(self._remove_spectra_annotation,
                          Input({"type": "remove-spectra-annotation-btn",
                                 "index": self._instance_index,
                                 "annotation_index": ALL},
                                "n_clicks"),
                          Output(self.annotations_panel.spectra_graph_annotations.id, 'children'),
                          app=app)

        # slice annotation removal callback
        targeted_callback(self._remove_slice_annotation,
                          Input({"type": "remove-slice-annotation-btn",
                                 "index": self._instance_index,
                                 "annotation_index": ALL},
                                "n_clicks"),
                          Output(self.annotations_panel.slice_graph_annotations.id, 'children'),
                          app=app)

    def make_layout(self):
        return html.Div(html.Div(self.components, className='row'),
                        className='container-fluid')  # , style={'flexGrow': 1})

    @cached_property
    def panels(self):
        return super(Viewer, self).panels + [self.annotations_panel]

    @property
    def map(self):
        """The currently displayed map slice at the current spectral index"""
        return self.graphs['map_graph'].map

    @property
    def spectrum(self):
        """The currently shown spectrum energy/wavenumber and intensity values"""
        return self.graphs['spectra_graph'].spectrum

    @property
    def spectral_value(self):
        """The current value of the crosshair position in energy/wavenumber"""
        return self.graphs['spectra_graph'].spectral_value

    @property
    def spectral_index(self):
        """The current index of the crosshair position along the energy/wavenumber domain"""
        return self.graphs['spectra_graph'].spectral_index

    @property
    def intensity(self):
        """The intensity value of the crosshair position"""
        return self.graphs['spectra_graph'].intensity

    @property
    def position(self):
        """The spatial position of the current spectrum"""
        return self.graphs['map_graph'].position

    @property
    def position_index(self):
        """The spatial position of the current spectrum as an index (y, x)"""
        return self.graphs['map_graph'].position_index

    @property
    def selection(self):
        """A mask array representing the current spatial selection region"""
        return self.graphs['map_graph'].selection

    @property
    def selection_indices(self):
        """The indices of all currently selected points, returned as (y, x)"""
        return self.graphs['map_graph'].selection_indices

    @property
    def spectra_annotations(self):
        """User-defined annotations on the spectra graph"""
        if 'spectra_graph' not in self.graphs:
            return []

        return self.graphs['spectra_graph'].annotations

    @property
    def slice_annotations(self):
        if not hasattr(self, 'map_graph'):
            return []
        return self.graphs['map_graph'].annotations

    def _add_spectra_annotation(self, annotation):
        annotation_index = next(self._annotation_counter)
        annotation['annotation_index'] = annotation_index
        self.graphs['spectra_graph'].add_annotation(annotation)

        annotation_content = f'{annotation["name"]}'
        if 'range' in annotation:
            r = annotation['range']
            annotation_content += f": {r[0]}, {r[1]} "
        elif 'position' in annotation:
            annotation_content += f": {annotation['position']} "

        # TODO add color

        btn = dbc.Button(html.I(className="fas fa-times"),
                         color='danger',
                         className='btn-sm',
                         n_clicks=0,
                         id={"type": "remove-spectra-annotation-btn",
                             "index": self._instance_index,
                             "annotation_index": annotation_index},
                         style={'marginLeft': 'auto'}
                         )

        item = dbc.NavItem(id={"type": "spectra-annotation-entry",
                               "index": self._instance_index,
                               "annotation_index": annotation_index},
                           className='annotation',
                           children=[dbc.NavLink(active=True, children=[
                               html.Span(children=annotation_content, className="annotation-content"),
                               btn
                           ])])

        self.annotations_panel.spectra_graph_annotations.children.append(item)

    def _remove_spectra_annotation(self, _):
        index = json.loads(dash.callback_context.triggered[0]["prop_id"].split(".")[0])['annotation_index']
        if index is not None:
            annotation_entries = self.annotations_panel.spectra_graph_annotations.children
            if annotation_entries is not None:
                # Remove the annotation entry (row) with matching annotation_index
                self.graphs['spectra_graph'].remove_annotation(index)
                spectra_annotations = filter(lambda annotation: annotation.id['annotation_index'] != index,
                                             annotation_entries)

                # Update the current spectra annotations list
                self.annotations_panel.spectra_graph_annotations.children = list(spectra_annotations)
        return self.annotations_panel.spectra_graph_annotations.children

    def _add_slice_annotation(self, annotation):
        annotation_index = next(self._annotation_counter)
        annotation['annotation_index'] = annotation_index

        for graph in self.graphs.values():
            if hasattr(graph, 'add_slice_annotation'):
                graph.add_slice_annotation(annotation)

        btn = dbc.Button(
            html.I(className="fas fa-times"),
            color='danger',
            className='btn-sm',
            n_clicks=0,
            id={"type": "remove-slice-annotation-btn",
                "index": self._instance_index,
                "annotation_index": annotation_index}
        )

        annotation_content = f'{annotation["name"]}: {annotation.get("color")} '
        item = dbc.NavItem(id={"type": "slice-annotation-entry",
                               "index": self._instance_index,
                               "annotation_index": annotation_index},
                           children=[dbc.NavLink(active=True, children=[
                               html.Span(children=annotation_content,
                                         className='annotation-content'),
                               btn
                           ])])

        self.annotations_panel.slice_graph_annotations.children.append(item)

    def _remove_slice_annotation(self, _):
        index = json.loads(dash.callback_context.triggered[0]["prop_id"].split(".")[0])['annotation_index']
        if index is not None:
            annotation_entries = self.annotations_panel.slice_graph_annotations.children
            if annotation_entries is not None:
                for graph in self.graphs.values():
                    if hasattr(graph, 'remove_slice_annotation'):
                        graph.remove_slice_annotation(index)

                slice_annotations = filter(lambda annotation: annotation.id['annotation_index'] != index,
                                           annotation_entries)

                # Update the current spectra annotations list
                self.annotations_panel.slice_graph_annotations.children = list(slice_annotations)
        return self.annotations_panel.slice_graph_annotations.children

    def _add_annotation_from_dialog(self, n_clicks):
        # Get the form input values
        input_states = dash.callback_context.states
        annotation = dict()
        if input_states['spectra-annotation-dialog-upper-bound.value'] is None:
            annotation['position'] = input_states['spectra-annotation-dialog-lower-bound.value']
        else:
            annotation['range'] = (input_states['spectra-annotation-dialog-lower-bound.value'],
                                   input_states['spectra-annotation-dialog-upper-bound.value'])

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

        return self.annotations_panel.spectra_graph_annotations.children

    def _add_slice_annotation_from_dialog(self, n_clicks):
        # Get the form input values
        input_states = dash.callback_context.states
        annotation = dict()
        annotation['name'] = input_states['slice-annotation-dialog-name.value']
        annotation['mask'] = self.graphs['map_graph']._selection_mask.z
        annotation['type'] = 'slice'

        # Color will come back as 'rgb': {'r': r, 'g': g, 'b': b, 'a': a},
        #     need to convert to plotly color: 'rgba(r,g,b,a)'
        color_picker = input_states['slice-annotation-dialog-color-picker.value']
        rgb = color_picker['rgb']
        color = f"rgb({rgb['r']}, {rgb['g']}, {rgb['b']})"
        annotation['color'] = color
        annotation['opacity'] = rgb['a']

        self._add_slice_annotation(annotation)

        return self.annotations_panel.slice_graph_annotations.children

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
