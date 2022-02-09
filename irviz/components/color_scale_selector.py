import json

import dash
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, MATCH, State, ALL

from ryujin.utils.dash import targeted_callback

__all__ = ['color_scales', 'ColorScaleSelector']

color_scales = ["Greys", "YlGnBu", "Greens", "YlOrRd", "Bluered", "RdBu", "Reds", "Blues", "Picnic", "Rainbow",
                "Portland", "Jet", "Hot", "Blackbody", "Earth", "Electric", "Viridis", "Cividis"]


def _noop(_, name):
    return name


class ColorScaleSelector(dbc.DropdownMenu):
    _callbacks_initialized = False

    def __init__(self, subtype, instance_index, value='Greys', values=None, app=None, **kwargs):
        """
        A DropDown which automatically swaps its label when an item is selected. This one is designed for use in selecting
        from a list of color scales. The above list of styles is not complete, and Dash/plotly support two key styles with
        different interpretations. Listen to its `label` property for callbacks to trigger on item selection.

        Parameters
        ----------
        app : dash.Dash
            A dash application instance, used for attaching callbacks
        _id : Union[str, dict]
            A unique Dash id
        value : str
            The initial value of the selector, expected to be present in values
        values : List[str]
            A list of options to select from

        """
        self.values = values
        if self.values is None:
            self.values = color_scales
        self.children = []
        _id = self._id(subtype, instance_index)
        for name in self.values:
            if isinstance(_id, dict):
                _id_copy = _id.copy()
                _id_copy['value'] = name
            else:
                _id_copy = _id + '-' + name
            self.children.append(dbc.DropdownMenuItem(name, id=_id_copy))

        kwargs['label'] = value
        kwargs['children'] = self.children
        kwargs['id'] = _id
        super(ColorScaleSelector, self).__init__(**kwargs)

        if app:
            self.init_callbacks(app)

    def init_callbacks(self, app):
        if not self._callbacks_initialized:
            targeted_callback(self.get_name,
                              Input({'type':'color-scale-selector',
                                     'subtype': MATCH,
                                     'index': MATCH,
                                     'value': ALL},
                                    'n_clicks'),
                              Output({'type':'color-scale-selector',
                                      'subtype': MATCH,
                                      'index': MATCH},
                                     'label'),
                              app=app,
                              prevent_initial_call=True
                              )
            ColorScaleSelector._callbacks_initialized = True

    def get_name(self, clicks, *_):
        return json.loads(dash.callback_context.triggered[0]['prop_id'].split('.')[0])['value']

    def _id(self, subtype, instance_index):
        _id = {'type':'color-scale-selector',
               'subtype': subtype,
               'index': instance_index
               }
        return _id
