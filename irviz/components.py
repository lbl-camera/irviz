from functools import partial
import dash_labs as dl
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from itertools import count
from dash.development.base_component import Component, _explicitize_args



from irviz.utils.dash import targeted_callback

color_scales = ["Greys", "YlGnBu", "Greens", "YlOrRd", "Bluered", "RdBu", "Reds", "Blues", "Picnic", "Rainbow",
                "Portland", "Jet", "Hot", "Blackbody", "Earth", "Electric", "Viridis", "Cividis"]


def _noop(_, name):
    return name


def ColorScaleSelector(app, _id, value='Greys', values=None):
    if values is None:
        values = color_scales
    children = []
    for name in values:
        _id_copy = _id.copy()
        _id_copy['value'] = name
        children.append(dbc.DropdownMenuItem(name, id=_id_copy))

    kwargs = dict(label=value,
                  children=children,
                  id=_id)
    selector = dbc.DropdownMenu(**kwargs)

    for name, item in zip(values, children):
        targeted_callback(partial(_noop, name=name),
                          Input(item.id, 'n_clicks'),
                          Output(selector.id, 'label'),
                          app=app,
                          )
    return selector
