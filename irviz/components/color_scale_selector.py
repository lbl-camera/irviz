from functools import partial

import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from methodtools import lru_cache

from ryujin.utils.dash import targeted_callback

__all__ = ['color_scales', 'ColorScaleSelector']

color_scales = ["Greys", "YlGnBu", "Greens", "YlOrRd", "Bluered", "RdBu", "Reds", "Blues", "Picnic", "Rainbow",
                "Portland", "Jet", "Hot", "Blackbody", "Earth", "Electric", "Viridis", "Cividis"]


def _noop(_, name):
    return name


class ColorScaleSelector(dbc.DropdownMenu):
    def __init__(self, _id, value='Greys', values=None, app=None):
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
        for name in self.values:
            if isinstance(_id, dict):
                _id_copy = _id.copy()
                _id_copy['value'] = name
            else:
                _id_copy = _id + '-' + name
            self.children.append(dbc.DropdownMenuItem(name, id=_id_copy))

        kwargs = dict(label=value,
                      children=self.children,
                      id=_id)
        super(ColorScaleSelector, self).__init__(**kwargs)

        if app:
            self.init_callbacks(app)

    @lru_cache(maxsize=1)
    def init_callbacks(self, app):
        for name, item in zip(self.values, self.children):
            targeted_callback(partial(_noop, name=name),
                              Input(item.id, 'n_clicks'),
                              Output(self.id, 'label'),
                              app=app,
                              prevent_initial_call=True
                              )

    @property
    def test(self):
        return True
