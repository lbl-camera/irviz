from typing import List, Callable

import dash_bootstrap_components as dbc
import dash_daq as daq
from dash.dependencies import Input, Output, State
from dash import html

from ryujin.utils.dash import targeted_callback

__all__ = ['SliceAnnotationDialog', 'SpectraAnnotationDialog', 'ModalDialog']


class ModalDialog(dbc.Modal):

    def __init__(self,
                 _id,
                 title: str,
                 children: List,
                 success_callback: Callable,
                 success_output: Output,
                 open_input: Input,
                 states: List[State], ):
        self.success_callback = success_callback
        self.success_output = success_output
        self.open_input = open_input
        self.states = states
        header = dbc.ModalHeader(title, id=f'{_id}-header')
        body = dbc.ModalBody(children, id=f'{_id}-body')
        self.cancel_button = dbc.Button("Cancel", id=f'{_id}-cancel', className="ml-auto", n_clicks=0)
        self.add_button = dbc.Button("Add", id=f'{_id}-add', className="ml-auto", n_clicks=0)
        footer = dbc.ModalFooter([self.add_button, self.cancel_button])

        super(ModalDialog, self).__init__([header, body, footer], id=_id)

    def init_callbacks(self, app):
        # Close dialog if Add/Cancel is clicked within it
        targeted_callback(lambda _: False,
                          Input(self.cancel_button.id, 'n_clicks'),
                          Output(self.id, 'is_open'),
                          app=app)
        targeted_callback(lambda _: False,
                          Input(self.add_button.id, 'n_clicks'),
                          Output(self.id, 'is_open'),
                          app=app)

        # When the Add button is clicked, trigger the success callback
        targeted_callback(self.success_callback,
                          Input(self.add_button.id, 'n_clicks'),
                          self.success_output,
                          *self.states,
                          app=app)

        # Open dialog when the open_input is triggered
        targeted_callback(lambda _: True,
                          self.open_input,
                          Output(self.id, 'is_open'),
                          app=app)


class SpectraAnnotationDialog(ModalDialog):
    def __init__(self, _id, **kwargs):
        name_input = dbc.Input(type="input", id=f'{_id}-name', placeholder="Enter annotation name", required=True)
        name_form = html.Div(
            [
                dbc.Label("Name"),
                name_input
                # dbc.FormText(
                #     "little text below the input component",
                #     color="secondary",
                # ),
            ], className='mb-3'
        )
        lower_bound_input = dbc.Input(type="number", id=f'{_id}-lower-bound', step=1, required=True)
        lower_bound_form = html.Div(
            [
                dbc.Label("Lower bound"),
                lower_bound_input
            ],
            id="styled-numeric-input",
            className='mb-3'
        )
        upper_bound_input = dbc.Input(type="number", id=f'{_id}-upper-bound', step=1, required=False)
        upper_bound_form = html.Div(
            [
                dbc.Label("Upper bound (optional)"),
                upper_bound_input
            ],
            className='mb-3'
        )

        color_input = daq.ColorPicker(id=f'{_id}-color-picker',
                                      label='Color Picker',
                                      value=dict(rgb=dict(r=100, g=200, b=200, a=.25)))

        super(SpectraAnnotationDialog, self).__init__(_id,
                                                      'Add Annotation',
                                                      [name_form, lower_bound_form, upper_bound_form, color_input],
                                                      states=[State(name_input.id, 'value'),
                                                              State(lower_bound_input.id, 'value'),
                                                              State(upper_bound_input.id, 'value'),
                                                              State(color_input.id, 'value')],
                                                      **kwargs)


class SliceAnnotationDialog(ModalDialog):
    def __init__(self, _id, **kwargs):
        name_input = dbc.Input(type="input", id=f'{_id}-name', placeholder="Enter annotation name", required=True)
        name_form = html.Div(
            [
                dbc.Label("Name"),
                name_input
                # dbc.FormText(
                #     "little text below the input component",
                #     color="secondary",
                # ),
            ],
            className='mb-3'
        )

        color_input = daq.ColorPicker(id=f'{_id}-color-picker',
                                      label='Color Picker',
                                      value=dict(rgb=dict(r=255, g=0, b=0, a=.3)))

        super(SliceAnnotationDialog, self).__init__(_id,
                                                    'Add Annotation',
                                                    [name_form, color_input],
                                                    states=[State(name_input.id, 'value'),
                                                            State(color_input.id, 'value')],
                                                    **kwargs)
