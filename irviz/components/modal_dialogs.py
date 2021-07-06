from typing import List, Any, Callable

import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

from irviz.utils.dash import targeted_callback

__all__ = ['spectra_annotation_dialog', 'modal_dialog']


def modal_dialog(app,
                 _id,
                 title: str,
                 children: List,
                 success_callback: Callable,
                 success_output: Output,
                 open_input: Input,
                 states: List[State],):
    header = dbc.ModalHeader(title, id=f'{_id}-header')
    body = dbc.ModalBody(children, id=f'{_id}-body')
    cancel_button = dbc.Button("Cancel", id=f'{_id}-cancel', className="ml-auto", n_clicks=0)
    add_button = dbc.Button("Add", id=f'{_id}-add', className="ml-auto", n_clicks=0)
    footer = dbc.ModalFooter([add_button, cancel_button])

    dialog = dbc.Modal([header, body, footer], id=_id)

    # Close dialog if Add/Cancel is clicked within it
    targeted_callback(lambda _: False,
                      Input(cancel_button.id, 'n_clicks'),
                      Output(dialog.id, 'is_open'),
                      app=app)
    targeted_callback(lambda _: False,
                      Input(add_button.id, 'n_clicks'),
                      Output(dialog.id, 'is_open'),
                      app=app)

    # When the Add button is clicked, trigger the success callback
    targeted_callback(success_callback,
                      Input(add_button.id, 'n_clicks'),
                      success_output,
                      *states,
                      app=app)

    # Open dialog when the open_input is triggered
    targeted_callback(lambda _: True,
                      open_input,
                      Output(dialog.id, 'is_open'),
                      app=app)

    return dialog


def spectra_annotation_dialog(app, _id, **kwargs):

    name_input = dbc.Input(type="input", id=f'{_id}-name', placeholder="Enter annotation name", required=True)
    name_form = dbc.FormGroup(
        [
            dbc.Label("Name"),
            name_input
            # dbc.FormText(
            #     "little text below the input component",
            #     color="secondary",
            # ),
        ]
    )
    lower_bound_input = dbc.Input(type="number", id=f'{_id}-lower-bound', step=1, required=True)
    lower_bound_form = dbc.FormGroup(
        [
            dbc.Label("Lower bound"),
            lower_bound_input
        ],
        id="styled-numeric-input",
    )
    upper_bound_input = dbc.Input(type="number", id=f'{_id}-upper-bound', step=1, required=False)
    upper_bound_form = dbc.FormGroup(
        [
            dbc.Label("Upper bound (optional)"),
            upper_bound_input
        ]
    )

    return modal_dialog(app,
                        _id,
                        'Add Annotation',
                        [name_form, lower_bound_form, upper_bound_form],
                        states=[State(name_input.id, 'value'),
                                State(lower_bound_input.id, 'value'),
                                State(upper_bound_input.id, 'value'),],
                        **kwargs)


def slice_annotation_dialog(app, _id, **kwargs):

    name_input = dbc.Input(type="input", id=f'{_id}-name', placeholder="Enter annotation name", required=True)
    name_form = dbc.FormGroup(
        [
            dbc.Label("Name"),
            name_input
            # dbc.FormText(
            #     "little text below the input component",
            #     color="secondary",
            # ),
        ]
    )

    return modal_dialog(app,
                        _id,
                        'Add Annotation',
                        [name_form],
                        states=[State(name_input.id, 'value')],
                        **kwargs)
