import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import geopandas as gpd
import shapely.geometry
import numpy as np
from jupyter_dash import JupyterDash

import plotly.graph_objects as go

import csv
import h5py
import astropy.time as astro_time

import time

from wmm import WMM

countries = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options
df_bar = pd.DataFrame({
    "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
    "Amount": [4, 1, 2, 2, 4, 5],
    "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
})

fig = px.bar(df_bar, x="Fruit", y="Amount", color="City", barmode="group")


class OAT:

    def __init__(self, fn, path = 'save_sim/', output_path = 'outputs/'):

        print('Initializing OAT simulation..')

        # set the font globally
        #plt.rcParams.update({'font.family':'sans-serif'})

        self.sim_path = path
        self.out_path = output_path

        f = h5py.File(self.sim_path + fn, 'r')
        print('Loading ' + fn + '..')

        self.dt = f.attrs['dt']

        self.X = f['states']['ECI']['X']
        self.Y = f['states']['ECI']['Y']
        self.Z = f['states']['ECI']['Z']
        self.LALN = f['states']['angular']['LALN']
        self.H = f['states']['ECI']['H']
        self.OE_ = f['states']['angular']['OE']

        self.B = f['B']['B']
        self.Bx = f['B']['Bx']
        self.By = f['B']['By']
        self.Bz = f['B']['Bz']

        self.times_jd = astro_time.Time(f['times']['JD'], format = 'jd').jd

        self.times_utc = astro_time.Time(f['times']['JD'], format = 'jd').datetime

        print('data successfully loaded..')

oat = OAT('test.hdf5')

step_size = 50

X = []
Y = []
Z = []

for index in range(1, len(oat.X), step_size):
    X.append(oat.X[index])
    Y.append(oat.Y[index])
    Z.append(oat.Z[index])


lats = []
lons = []

for index in range(1, len(oat.LALN), step_size):
    lats.append(oat.LALN[index,0])
    lons.append(oat.LALN[index,1])
    print("loading lat/lon #" + str(index))


fig1 = go.Figure(
    data=[go.Scatter3d(x=X, y=Y, z=Z,
                     mode="lines",
                     line=dict(width=2, color="blue")),
          go.Scatter3d(x=X, y=Y, z=Z,
                     mode="lines",
                     line=dict(width=2, color="blue"))],
    layout=go.Layout(
        title_text="Kinematic Generation of a Planar Curve", hovermode="closest",
        updatemenus=[dict(type="buttons",
                          buttons=[dict(label="Play",
                                        method="animate",
                                        args=[None])])]),
    frames=[go.Frame(
        data=[go.Scatter3d(
            x=[X[k]],
            y=[Y[k]],
            z=[Z[k]],
            mode="markers",
            marker=dict(color="red", size=10))])

        for k in range(len(X))]
)



# https://stackoverflow.com/questions/63877348/how-do-i-set-dot-sizes-and-colors-for-a-plotly-express-scatter-geo

fig2 = go.Figure(
    data=[go.Scattergeo(lon=lons, lat=lats,
                     mode="lines",
                     line=dict(width=2, color="blue")),
          go.Scattergeo(lon=lons, lat=lats,
                     mode="lines",
                     line=dict(width=2, color="blue"))],
    layout=go.Layout(
        title_text="Kinematic Generation of a Planar Curve", hovermode="closest",
        updatemenus=[dict(type="buttons",
                          buttons=[dict(label="Play",
                                        method="animate",
                                        args=[None])])]),
    frames=[go.Frame(
        data=[go.Scattergeo(
            lon=[lons[k]],
            lat=[lats[k]],
            mode="markers",
            marker=dict(color="red", size=10))])

        for k in range(len(lats))]
)






fig3 = go.Figure(
    data=[go.Scatter(x=oat.times_utc, y=oat.B, name='|B|'),
          go.Scatter(x=oat.times_utc, y=oat.Bx, name='Bx'),
          go.Scatter(x=oat.times_utc, y=oat.By, name='By'),
          go.Scatter(x=oat.times_utc, y=oat.Bz, name='Bz')]
)


times = [oat.times_utc[0]]
B = [oat.B[0]]
Bx = [oat.Bx[0]]
By = [oat.By[0]]
Bz = [oat.Bz[0]]
for index in range(1, len(oat.times_utc), step_size):
    times.append(oat.times_utc[index])
    B.append(oat.B[index])
    Bx.append(oat.Bx[index])
    By.append(oat.By[index])
    Bz.append(oat.Bz[index])
    print("loading B #" + str(index))

fig4 = go.Figure(
    data=[go.Scatter(x=[times[0]], y=[B[0]], name='|B|'),
          go.Scatter(x=[times[0]], y=[Bx[0]], name='Bx'),
          go.Scatter(x=[times[0]], y=[By[0]], name='By'),
          go.Scatter(x=[times[0]], y=[Bz[0]], name='Bz')],
    layout=go.Layout(
        yaxis=dict(range=[-60,60], autorange=False),
        title="Start Title",
        updatemenus=[dict(
            type="buttons",
            buttons=[dict(label="Play",
                          method="animate",
                          args=[None])])]
    ),
    frames=[go.Frame(
        data=[
          go.Scatter(x=times[:k], y=B[:k], name='|B|'),
          go.Scatter(x=times[:k], y=Bx[:k], name='Bx'),
          go.Scatter(x=times[:k], y=By[:k], name='By'),
          go.Scatter(x=times[:k], y=Bz[:k], name='Bz')])

        for k in range(1, len(times))]
)



app.layout = html.Div(children=[
    # All elements from the top of the page
    html.Div([
        html.Div([
            html.H1(children='Hello Dash'),

            html.Div(children='''
                Dash: A web application framework for Python.
            '''),

            dcc.Graph(
                id='graph1',
                figure=fig1
            ),  
        ], className='six columns'),
        html.Div([
            html.H1(children='Hello Dash'),

            html.Div(children='''
                Dash: A web application framework for Python.
            '''),

            dcc.Graph(
                id='graph2',
                figure=fig2
            ),  
        ], className='six columns'),
    ], className='row'),
    # New Div for all elements in the new 'row' of the page
    html.Div([
        html.Div([
            html.H1(children='Hello Dash'),

            html.Div(children='''
                Dash: A web application framework for Python.
            '''),

            dcc.Graph(
                id='graph3',
                figure=fig
            ),  
        ], className='six columns'),
        html.Div([
            html.H1(children='Hello Dash'),

            html.Div(children='''
                Dash: A web application framework for Python.
            '''),

            dcc.Graph(
                id='graph4',
                figure=fig4
            ),  
        ], className='six columns'),
    ], className='row'),
])

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)