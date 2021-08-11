import sys
import re
from typing import Callable
# noinspection PyUnresolvedReferences
from inspect import signature, _empty

import dash
import dash_table
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_core_components as dcc
from dash.dependencies import Input, ALL, Output, State

from ryujin.utils import targeted_callback

"""
{'name', 'title', 'value', 'type', 
"""


class SimpleItem(dbc.FormGroup):
    def __init__(self, name, base_id, title=None, value=None, min=None, max=None, step=None, type='number', debounce=True):
        self.name = name

        self.label = dbc.Label(title or name)
        self.input = dbc.Input(value=value,
                               min=min,
                               max=max,
                               step=step,
                               type=type,
                               debounce=debounce,
                               id={'type': 'kwarg_editor_item',
                                   'base_id': base_id,
                                   'name': name})

        super(SimpleItem, self).__init__(children=[self.label, self.input])


class FloatItem(SimpleItem):
    pass


class IntItem(SimpleItem):
    def __init__(self, *args, **kwargs):
        if 'min' not in kwargs:
            kwargs['min'] = -sys.maxsize - 1  # min must be set for int validation to be enabled
        super(IntItem, self).__init__(*args, step=1, **kwargs)


class StrItem(SimpleItem):
    def __init__(self, *args, **kwargs):
        super(StrItem, self).__init__(*args, type='text', **kwargs)


class ParameterEditor(dbc.Form):

    type_map = {float: FloatItem,
                int: IntItem,
                str: StrItem,
                }

    def __init__(self, _id, parameters, **kwargs):
        self._parameters = parameters

        super(ParameterEditor, self).__init__(id=_id, children=[], className='kwarg-editor', **kwargs)
        self.children = self.build_children()

    def init_callbacks(self, app):
        targeted_callback(self.stash_value,
                          Input({'type': 'kwarg_editor_item',
                                 'base_id': self.id,
                                 'name': ALL},
                                'value'),
                          Output(self.id, 'n_submit'),
                          State(self.id, 'n_submit'),
                          app=app)

    def stash_value(self, value):
        # find the changed item name from regex
        r = '(?<=\"name\"\:\")\w+(?=\")'
        matches = re.findall(r, dash.callback_context.triggered[0]['prop_id'])

        if not matches:
            raise LookupError('Could not find changed item name. Check that all parameter names use simple chars (\\w)')

        name = matches[0]
        self.parameters[name]['value'] = value

        print(self.values)

        return next(iter(dash.callback_context.states.values())) or 0 + 1

    @property
    def values(self):
        return {param['name']: param.get('value', None) for param in self._parameters}

    @property
    def parameters(self):
        return {param['name']: param for param in self._parameters}

    def _determine_type(self, parameter_dict):
        if 'type' in parameter_dict:
            if parameter_dict['type'] in self.type_map:
                return parameter_dict['type']
            elif parameter_dict['type'].__name__ in self.type_map:
                return parameter_dict['type'].__name__
        elif type(parameter_dict['value']) in self.type_map:
            return type(parameter_dict['value'])
        raise TypeError(f'No item type could be determined for this parameter: {parameter_dict}')

    def build_children(self, values=None):
        children = []
        for parameter_dict in self._parameters:
            parameter_dict = parameter_dict.copy()
            if values and parameter_dict['name'] in values:
                parameter_dict['value'] = values[parameter_dict['name']]
            type = self._determine_type(parameter_dict)
            parameter_dict.pop('type', None)
            item = self.type_map[type](**parameter_dict, base_id=self.id)
            children.append(item)

        return children


class KwargsEditor(ParameterEditor):
    def __init__(self, _id, func: Callable, **kwargs):

        parameters = [{'name': name, 'value': param.default} for name, param in signature(func).parameters.items()
                      if param.default is not _empty]

        super(KwargsEditor, self).__init__(_id, parameters=parameters, **kwargs)


if __name__ == '__main__':

    import dash
    import dash_bootstrap_components as dbc

    app_kwargs = {'external_stylesheets': [dbc.themes.BOOTSTRAP]}
    app = dash.Dash(__name__, **app_kwargs)

    item_list = ParameterEditor(_id='params', parameters=[{'name': 'test', 'value': 2},
                                                          {'name': 'test2', 'value': 'blah'},
                                                          {'name': 'test3', 'value': 3.2, 'type': float}])

    def my_func(a, b, c=1, d='blah', e=23.4):
        ...

    kwarg_list = KwargsEditor(_id='kwargs', func=my_func)

    kwarg_list.init_callbacks(app)
    item_list.init_callbacks(app)

    app.layout = html.Div([item_list, kwarg_list])
    app.run_server(debug=True)
