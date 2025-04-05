import requests
import plotly.graph_objects as go

import dash
import dash_core_components as dcc
import dash_html_components as html

latitude = -22.2974
longitude = -46.6062

#making the plot
fig = go.Figure(go.Scattergeo(lat=[latitude], lon=[longitude])) #if you are passing just one lat and lon, put it within "[]""

#editing the marker
fig.update_traces(marker_size=20, line=dict(color='Red'))

# this projection_type = 'orthographic is the projection which return 3d globe map'
fig.update_geos(projection_type="orthographic") 

app = dash.Dash()
app.layout = html.Div(children=[
    html.H1(children='Hello Dash'),

    html.Div(children='''
        Dash: A web application framework for your data.
    '''),

    dcc.Graph(
        id='example-graph',
        figure=fig
    )
])

app.run_server(debug=True, port=8044)