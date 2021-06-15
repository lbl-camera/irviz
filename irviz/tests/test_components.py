from dash.dependencies import Input, Output
import dash
import dash_bootstrap_components as dbc
import dash_html_components as html


from irviz.components import ColorScaleSelector
from irviz.utils.dash import targeted_callback


def test_color_scale_selector(dash_duo):
    app_kwargs = {'external_stylesheets': [dbc.themes.BOOTSTRAP]}
    app = dash.Dash(__name__, **app_kwargs)

    c = ColorScaleSelector(app, 'blah')

    app.layout = html.Div([c])
    dash_duo.start_server(app)

    button_a = dash_duo.wait_for_element_by_css_selector('#A', timeout=100000000000000000)
