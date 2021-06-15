from itertools import count
import re
import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import plotly.graph_objects as go
from dash.dependencies import Input, Output, ALL
from dash.exceptions import PreventUpdate
from dash.development.base_component import Component
from irviz.utils.dash import targeted_callback


transparent_color_scales = {'TransparentRed': [[0, "rgba(255, 0, 0, 0)"],
                                               [1, "rgba(255, 0, 0, 255)"]],
                            'TransparentGreen': [[0, "rgba(0, 255, 0, 0)"],
                                                [1, "rgba(0, 255, 0, 255)"]],
                            'TransparentBlue': [[0, "rgba(0, 0, 255, 0)"],
                                                [1, "rgba(0, 0, 255, 255)"]],
                            'TransparentYellow': [[0, "rgba(255, 255, 0, 0)"],
                                                  [1, "rgba(255, 255, 0, 255)"]],
                            'TransparentOrange': [[0, "rgba(255, 69, 0, 0)"],
                                                  [1, "rgba(255, 69, 0, 255)"]],
                            'TransparentPurple': [[0, "rgba(255, 0, 255, 0)"],
                                                  [1, "rgba(255, 0, 255, 255)"]],
                            'TransparentCyan': [[0, "rgba(0, 255, 255, 0)"],
                                                [1, "rgba(0, 255, 255, 255)"]]
                            }


decomposition_color_scales = ['gray']+list(transparent_color_scales.keys())


def nearest_bin(x, bounds, bin_count):
    return int((x-bounds[0])/(bounds[1]-bounds[0])*bin_count)


class SpectraPlotGraph(dcc.Graph):
    _counter = count(0)

    title = 'Spectra Intensities'

    def __init__(self, data, bounds, parent, **kwargs):
        """Interactive Graph that shows spectral intensities at a selectable energy / wave-number index.

        Parameters
        ----------
        data : dask array
            3D array containing data with axes E (or wave-number), y, and x for displaying in the Graph
        bounds : ndarray-like
            Collection that defines the bounds (min & max) pairs for E / Wave-number, y, and x data
            (e.g. a list that contains 3 min/max pairs)
        parent : Component
            Reference to Component that created this Graph (for registering callbacks)
        """
        self._instance_index = next(self._counter)
        self._data = data
        self._parent = parent
        self._bounds = bounds

        self.xaxis_title = kwargs.pop('xaxis_title', '')
        self.yaxis_title = kwargs.pop('yaxis_title', '')

        #  default to middle x,y
        _y_index = (self._data.shape[1] - 1) // 2
        _x_index = (self._data.shape[2] - 1) // 2

        y = np.asarray(self._data[:, _y_index, _x_index])
        x = np.linspace(bounds[0][0], bounds[0][1], self._data.shape[0])
        self._plot = go.Scattergl(x=x, y=y)

        # Define starting point for energy index (for the slicer line trace)
        default_slice_index = (bounds[0][1] + bounds[0][0]) / 2  # estimate
        # Find the closest wavenumber / energy value to use
        default_slice_index = x[np.abs(np.array(x) - default_slice_index).argmin()]

        # x coords positioned relative to the x-axis values
        # y coords positioned according to the plot height (0 = bottom, 1.0 = top)
        self._energy_line = go.layout.Shape(type='line',
                                            # width=3,
                                            xref='x',
                                            yref='paper',
                                            x0=default_slice_index,
                                            x1=default_slice_index,
                                            y0=0,
                                            y1=1)

        fig = self._update_figure()

        super(SpectraPlotGraph, self).__init__(id=self._id(),
                                               figure=fig,
                                               className='col-lg-12')

    def register_callbacks(self):
        self._parent._app.callback(
            Output(self.id, 'figure'),
            Input(self._parent.map_graph.id, 'clickData'),
            Input(self.id, 'clickData'),
            Input(self._parent.decomposition_graph.id, 'clickData')
        )(self._show_plot)

        # Wire-up visibility toggle
        self._parent._app.callback(
            Output(self.id, 'style'),
            Input(self._parent.graph_toggles.id, 'value')
        )(self._set_visibility)
        
    @property
    def spectrum(self):
        """The currently shown spectrum energy/wavenumber and intensity values"""
        return self._plot.x, self._plot.y

    @property
    def spectral_value(self):
        """The current value of the crosshair position in energy/wavenumber"""
        return self._energy_line.x0
        
    @property
    def spectral_index(self):
        """The current index of the crosshair position along the energy/wavenumber domain"""
        return self._plot.x.tolist().index(self._energy_line.x0)

    @property
    def intensity(self):
        """The intensity value of the crosshair position"""
        intensity_index = self._plot.x.tolist().index(self._energy_line.x0)
        return self._plot.y[intensity_index]
    
    @property
    def position(self):
        """The spatial position of the current spectrum"""
        return self._parent.map_graph.position

    @staticmethod
    def _set_visibility(switches_value):
        if 'show_spectra' in switches_value:
            return {'display': 'block'}
        else:
            return {'display': 'none'}

    def _update_figure(self):
        fig = go.Figure(self._plot)
        fig.update_layout(title=self.title,
                          xaxis_title=self.xaxis_title,
                          yaxis_title=self.yaxis_title)
        fig.add_shape(self._energy_line)
        return fig

    def _show_plot(self, slice_graph_click_data, self_click_data, decomposition_click_data):
        triggered = dash.callback_context.triggered
        if not triggered:
            raise PreventUpdate

        # When the slice graph is clicked, update plot with the clicked x,y coord
        if self._parent.map_graph.id in triggered[0]['prop_id']:
            self._show_click(slice_graph_click_data)

        elif self._parent.decomposition_graph.id in triggered[0]['prop_id']:
            self._show_click(decomposition_click_data)

        # When this SpectraGraph itself is clicked, update the energy slicer line
        elif self_click_data is not None:
            e = self_click_data["points"][0]["x"]
            self._energy_line.x0 = e
            self._energy_line.x1 = e

        return self._update_figure()

    def _show_click(self, click_data):
        y = click_data["points"][0]["y"]
        x = click_data["points"][0]["x"]
        _x_index = nearest_bin(x, self._bounds[2], self._data.shape[2])
        _y_index = nearest_bin(y, self._bounds[1], self._data.shape[1])
        self._plot.y = np.asarray(self._data[:, _y_index, _x_index])

    def _id(self):
        return f'spectraplot_{self._instance_index}'


class SliceGraph(dcc.Graph):
    """Dash Graph for viewing 2D slices of 3D data.

    Parameters
    ----------
    data : dask.array
        3D data array
    parent : html.Div
        The parent object that creates this Graph

    """
    _counter = count(0)

    title = ''
    aspect_locked = True

    def __init__(self, data, bounds, parent, slice_axis=0, traces=None, shapes=None, **kwargs):

        # Cache our data and parent for use in the callbacks
        self._data = data
        self._parent = parent
        self._bounds = bounds
        self._instance_index = next(self._counter)
        self._traces = traces or []
        self._shapes = shapes or []

        self.xaxis_title = kwargs.pop('xaxis_title', '')
        self.yaxis_title = kwargs.pop('yaxis_title', '')

        self._h_line = go.layout.Shape(type='line',
                                       # width=3,
                                       xref='paper',
                                       yref='y',
                                       x0=0,
                                       x1=1,
                                       y0=(bounds[1][1]+bounds[1][0])/2,
                                       y1=(bounds[1][1]+bounds[1][0])/2)
        self._v_line = go.layout.Shape(type='line',
                                       xref='x',
                                       yref='paper',
                                       x0=(bounds[2][1]+bounds[2][0])/2,
                                       x1=(bounds[2][1]+bounds[2][0])/2,
                                       y0=0,
                                       y1=1)

        self._shapes.extend([self._h_line, self._v_line])

        figure = self._update_figure()
        super(SliceGraph, self).__init__(figure=figure,
                                         id=self._id(),
                                         className='col-lg-4', **kwargs)

    def _id(self):
        return f'slicegraph_{self._instance_index}'

    def register_callbacks(self):
        ...

    def _update_figure(self):
        fig = go.Figure(self._traces)
        fig.update_layout(title=self.title,
                          xaxis_title=self.xaxis_title,
                          yaxis_title=self.yaxis_title)
        if self.aspect_locked:
            fig.update_yaxes(scaleanchor="x", scaleratio=1)
        for shape in self._shapes:
            fig.add_shape(shape)
        return fig

    def show_click(self, click_data):
        y_index = click_data["points"][0]["y"]
        x_index = click_data["points"][0]["x"]
        self._h_line.y0 = y_index
        self._h_line.y1 = y_index
        self._v_line.x0 = x_index
        self._v_line.x1 = x_index

        return self._update_figure()

    @property
    def position(self):
        """The current spatial position of the crosshair"""
        return self._v_line.x0, self._h_line.y0


class MapGraph(SliceGraph):
    """Dash Graph for viewing 2D slices of 3D data.

    Parameters
    ----------
    data : dask.array
        3D data array
    parent : html.Div
        The parent object that creates this Graph

    """
    title = 'IR Spectral Map'

    def __init__(self, data, bounds, parent, slice_axis=0, traces=None, shapes=None, **kwargs):

        default_slice_index = (data.shape[0] - 1) // 2

        # Create traces (i.e. 'glyphs') that will comprise a plotly Figure
        self._image = go.Heatmap(z=np.asarray(data[default_slice_index]),
                                 colorscale='gray',
                                 y0=bounds[1][0],
                                 dy=(bounds[1][1]-bounds[1][0])/data.shape[1],
                                 x0=bounds[2][0],
                                 dx=(bounds[2][1]-bounds[2][0])/data.shape[2],
                                 )
        self._selection_mask = go.Heatmap(z=np.ones(data[0].shape) * np.NaN,
                                          colorscale='reds',
                                          opacity=0.3,
                                          showscale=False,
                                          y0=bounds[1][0],
                                          dy=(bounds[1][1]-bounds[1][0])/data.shape[1],
                                          x0=bounds[2][0],
                                          dx=(bounds[2][1]-bounds[2][0])/data.shape[2])
        x, y = np.meshgrid(np.linspace(bounds[2][0], bounds[2][1], data.shape[2], endpoint=False),
                           np.linspace(bounds[1][0], bounds[1][1], data.shape[1], endpoint=False))

        # This dummy scatter trace is added to support lasso selection
        self._dummy_scatter = go.Scattergl(x=x.ravel(),
                                           y=y.ravel(),
                                           mode='markers',
                                           marker={'color': 'rgba(0,0,0,0)'}
                                           )
        traces = (traces or []) + [self._image, self._selection_mask, self._dummy_scatter]

        super(MapGraph, self).__init__(data,
                                       bounds,
                                       parent,
                                       slice_axis=slice_axis,
                                       traces=traces,
                                       shapes=shapes,
                                       **kwargs
                                       # config={'modeBarButtonsToAdd': ['lasso2d']}
                                       )

    def register_callbacks(self):
        # When the spectra graph is clicked, update image slicing
        targeted_callback(self.update_slice,
                          Input(self._parent.spectra_graph.id, 'clickData'),
                          Output(self.id, 'figure'),
                          app=self._parent._app)

        # When this SliceGraph itself is clicked, update its x,y slicer lines
        targeted_callback(self.show_click,
                          Input(self.id, 'clickData'),
                          Output(self.id, 'figure'),
                          app=self._parent._app)

        # When the decomposition graph is clicked show the same position
        targeted_callback(self.show_click,
                          Input(self._parent.decomposition_graph.id, 'clickData'),
                          Output(self.id, 'figure'),
                          app=self._parent._app)

        # When points are selected in the pair plot, show them here
        targeted_callback(self._show_selection_mask,
                          Input(self._parent.pair_plot_graph.id, 'selectedData'),
                          Output(self.id, 'figure'),
                          app=self._parent._app)

        # When this SliceGraph is lasso'd, update the selection mask
        targeted_callback(self._show_selection_mask,
                          Input(self.id, 'selectedData'),
                          Output(self.id, 'figure'),
                          app=self._parent._app)

    def update_slice(self, spectra_graph_click_data):
        slice = spectra_graph_click_data["points"][0]["x"]
        slice_index = nearest_bin(slice, self._bounds[0], self._data.shape[0])
        self._image.z = np.asarray(self._data[slice_index])

        return self._update_figure()

    def _show_selection_mask(self, selection):
        # Get x,y from the raveled indexes
        raveled_indexes = list(map(lambda point: point['pointIndex'], selection['points']))
        mask = np.zeros(self._data[0].shape)
        # Cannot be 0s - must be NaNs (eval to None) so it doesn't affect underlying HeatMap
        mask.fill(np.NaN)
        mask.ravel()[raveled_indexes] = 1
        # Create overlay
        self._selection_mask.z = mask

        return self._update_figure()


class DecompositionGraph(SliceGraph):
    title = 'Decomposition Maps'

    def __init__(self, data, bounds, parent, *args, **kwargs):

        traces = []
        for i in range(data.shape[0]):
            color_scale = decomposition_color_scales[i % len(decomposition_color_scales)]
            color_scale = transparent_color_scales.get(color_scale, color_scale)

            traces.append(go.Heatmap(z=np.asarray(data[i]),
                                 colorscale=color_scale,
                                 y0=bounds[1][0],
                                 dy=(bounds[1][1]-bounds[1][0])/data.shape[1],
                                 x0=bounds[2][0],
                                 dx=(bounds[2][1]-bounds[2][0])/data.shape[2],
                                 visible=(i==0),
                                 opacity=.5 if i else 1,
                                 ))

        kwargs['traces'] = traces

        super(DecompositionGraph, self).__init__(data, bounds, parent, *args, **kwargs)

    def register_callbacks(self):
        # Set up callbacks
        # ----------------

        # Wire-up visibility toggle
        self._parent._app.callback(
            Output(self.id, 'style'),
            Input(self._parent.graph_toggles.id, 'value')
        )(self._set_visibility)

        # Wire-up opacity sliders
        targeted_callback(self.set_component_opacity,
                          Input({'type': 'component-opacity', 'index': ALL}, 'value'),
                          Output(self.id, 'figure'),
                          app=self._parent._app)
        
        # Disable sliders when their component is hidden
        targeted_callback(self.disable_sliders,
                          Input(self._parent.decomposition_component_selector.id, 'value'),
                          Output(self._parent.component_opacity_sliders.id, 'children'),
                          app=self._parent._app)

        # Show components when selected
        targeted_callback(self.show_components,
                          Input(self._parent.decomposition_component_selector.id, 'value'),
                          Output(self.id, 'figure'),
                          app=self._parent._app)

        # Show clicked position when this graph is clicked
        targeted_callback(self.show_click,
                          Input(self.id, 'clickData'),
                          Output(self.id, 'figure'),
                          app=self._parent._app)

        # Show clicked position when map graph is clicked
        targeted_callback(self.show_click,
                          Input(self._parent.map_graph.id, 'clickData'),
                          Output(self.id, 'figure'),
                          app=self._parent._app)

        # Update the color scale when new item is selected
        targeted_callback(self.set_color_scale,
                          Input({'type':'color-scale-selector', 'index': ALL}, 'label'),
                          Output(self.id, 'figure'),
                          app=self._parent._app)

    def set_color_scale(self, color_scale):
        i = int(re.findall('(?<="index":)\\d+(?=,)', dash.callback_context.triggered[0]['prop_id'])[0])
        color_scale = transparent_color_scales.get(color_scale, color_scale)
        self._traces[i].colorscale = color_scale

        return self._update_figure()

    def _opacity_slider(self, i):
        return self._parent.component_opacity_sliders.children[i]

    def _color_scale_selector(self, i):
        return self._parent.component_color_scale_selectors.children[i]

    def set_component_opacity(self, value):
        i = int(re.findall('(?<="index":)\\d+(?=,)', dash.callback_context.triggered[0]['prop_id'])[0])
        self._opacity_slider(i).value = value
        self._update_opacity()
        return self._update_figure()

    def _update_opacity(self):
        # Get a sum of all enabled slider values minus the first enabled value
        total = 0
        for slider in self._parent.component_opacity_sliders.children:
            if not slider.disabled:
                total += slider.value

        # Set each trace's opacity to a value proportional to its weight; always set first visible trace's opacity to 1
        bg_set = False
        for i, trace in enumerate(self._traces):
            if trace.visible:
                if not bg_set:
                    trace.opacity = 1
                    bg_set = True
                    continue

                trace.opacity = self._opacity_slider(i).value / total

    def show_components(self, component_indices):
        for i, trace in enumerate(self._traces):
            trace.visible = (i in component_indices)
            trace.showscale = len(component_indices) < 2
            self._opacity_slider(i).disabled = not (i in component_indices)  # TODO: set this in a separate callback that outputs to the slider

        self._update_opacity()

        return self._update_figure()
    
    def disable_sliders(self, component_indices):
        for i, trace in enumerate(self._traces):
            self._opacity_slider(i).disabled = not (i in component_indices)  # TODO: set this in a separate callback that outputs to the slider

        return self._parent.component_opacity_sliders.children
        

    @staticmethod
    def _set_visibility(switches_value):
        if 'show_decomposition' in switches_value:
            return {'display':'block'}
        else:
            return {'display':'none'}


class PairPlotGraph(dcc.Graph):
    _counter = count(0)
    title = 'Pair Plot'

    def __init__(self, data, parent):
        self._instance_index = next(self._counter)

        # Cache our data and parent for use in the callbacks
        self._data = data
        self._parent = parent
        self._component1 = self._component2 = 0

        # Create traces (i.e. 'glyphs') that will comprise a plotly Figure
        self._scatter = go.Scattergl(x=[], y=[], mode='markers')

        figure = self._update_figure()
        super(PairPlotGraph, self).__init__(figure=figure,
                                            id=f'pair_plot_{self._instance_index}',
                                            className='col-lg-4')

    def register_callbacks(self):
        # Set up callbacks
        # ----------------

        # When the parent viewer's 'spectra_graph' is clicked
        #     we need to update the internal Figure for this Graph
        # When MapGraph is lasso'd, show that selection here too
        # Note: this can't be a targeted callback, since multiple values are required
        self._parent._app.callback(
            Output(self.id, 'figure'),
            Input(self._parent.decomposition_component_1.id, 'value'),
            Input(self._parent.decomposition_component_2.id, 'value'),
            Input(self._parent.map_graph.id, 'selectedData')
        )(self.show_pair_plot)

        # Set up selection tool callbacks
        targeted_callback(self._show_selection_info,
                          Input(self.id, 'selectedData'),
                          Output(self._parent.info_content.id, 'children'),
                          app=self._parent._app)

        # Wire-up visibility toggle
        targeted_callback(self._set_visibility,
                          Input(self._parent.graph_toggles.id, 'value'),
                          Output(self.id, 'style'),
                          app=self._parent._app)

    def _show_selection_info(self, selected_data):
        if not selected_data:
            return "info"
        return str(list(map(lambda point: point['pointIndex'], selected_data['points'])))

    def _update_figure(self):
        """ Remake the figure to force a display update """
        fig = go.Figure([self._scatter])
        fig.update_layout(title=self.title,
                          xaxis_title=f'Component #{self._component1+1}',
                          yaxis_title=f'Component #{self._component2+1}')
        return fig

    def show_pair_plot(self, component1, component2, selectedData):
        if component1 is None or component2 is None:
            raise PreventUpdate

        triggered = dash.callback_context.triggered
        if self._parent.map_graph.id in triggered[0]['prop_id']:
            raveled_indexes = list(map(lambda point: point['pointIndex'], selectedData['points']))
            self._scatter.selectedpoints = raveled_indexes

        x = self._data[component1]
        y = self._data[component2]
        self._scatter.x = np.asarray(x.ravel())
        self._scatter.y = np.asarray(y.ravel())
        self._component1 = component1
        self._component2 = component2

        return self._update_figure()

    @staticmethod
    def _set_visibility(switches_value):
        if 'show_pair_plot' in switches_value:
            return {'display':'block'}
        else:
            return {'display':'none'}
