from typing import Union, Callable
from functools import lru_cache
from dash._utils import create_callback_id
from dash.dependencies import handle_callback_args
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
import dash
from irviz import app
from dataclasses import dataclass


_targeted_callbacks = []


@dataclass
class Callback:
    input: Input
    output: Output
    callable: Callable


def _dispatcher(*_):
    triggered = dash.callback_context.triggered
    if not triggered:
        raise PreventUpdate

    for callback in _targeted_callbacks:
        if create_callback_id(callback.input) == triggered[0]['prop_id'] and \
                create_callback_id(callback.output) == \
                f"{dash.callback_context.outputs_list['id']}.{dash.callback_context.outputs_list['property']}":
            return callback.callable(triggered[0]['value'])


def targeted_callback(callback, input:Input, output:Output, app=app):
    callback_id = create_callback_id(output)
    if callback_id in app.callback_map:
        for callback_spec in app._callback_list:
            if callback_spec['output'] == callback_id:
                callback_spec['inputs'].append(input.to_dict())
    else:
        app.callback(output, input)(_dispatcher)
    _targeted_callbacks.append(Callback(input, output, callback))
