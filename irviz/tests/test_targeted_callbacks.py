from dash.dependencies import Input, Output
import dash
import dash_bootstrap_components as dbc
import dash_html_components as html
from ryujin.utils.dash import targeted_callback


# For reference, here is a comparable (albeit simple) example with standard callbacks

# class Test(html.Div):
#     def __init__(self):
#         self.button_a = dbc.Button('A', id='A', value='A')
#         self.button_b = dbc.Button('B', id='B', value='B')
#         self.div = html.Div(id='D')
#         super(Test, self).__init__(children=[self.button_a, self.button_b, self.div])
#
#         irviz.app.callback(Output('D', 'children'),
#                            Input('A', 'n_clicks'),
#                            Input('B', 'n_clicks'),
#                            prevent_initial_call=True)(self.callback)
#
#
#     def callback(self, input1, input2):
#         triggered = dash.callback_context.triggered
#         print(irviz.app._callback_list)
#         return 'Pressed ?', input1, input2, str(irviz.app.callback_map), str(irviz.app._callback_list)


def test_targeted_callbacks(dash_duo):
    app = dash.Dash(__name__)

    class Test(html.Div):
        def __init__(self):
            self.button_a = dbc.Button('A', id='A', value='A')
            self.button_b = dbc.Button('B', id='B', value='B')
            self.div = html.Div(id='D')
            super(Test, self).__init__(children=[self.button_a, self.button_b, self.div])

            targeted_callback(self.callback1, Input('A', 'n_clicks'), Output('D','children'), app=app)
            targeted_callback(self.callback2, Input('B', 'n_clicks'), Output('D','children'), app=app)

        def callback1(self, input1):
            return 'Pressed A'

        def callback2(self, input2):
            return 'Pressed B'

    test = Test()
    app.layout = test
    dash_duo.start_server(app)

    button_a = dash_duo.wait_for_element_by_css_selector('#A')
    button_b = dash_duo.wait_for_element_by_css_selector('#B')
    button_a.click()
    assert dash_duo.wait_for_text_to_equal('#D', 'Pressed A')
    button_b.click()
    assert dash_duo.wait_for_text_to_equal('#D', 'Pressed B')
