import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
from jupyter_dash import JupyterDash

import plotly.graph_objects as go

import csv
import h5py
import geopandas as gpd
import astropy.time as astro_time

import time

from wmm import WMM

countries = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options



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

    def __earth_3d(self):

        R_e = 6371 #km

        u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:100j]
        x = (np.cos(u)*np.sin(v))*R_e
        y = (np.sin(u)*np.sin(v))*R_e
        z = (np.cos(v))*R_e

        earth = np.array([x, y, z])

        return earth 

oat = OAT('test.hdf5')


step_size = 50
X = []
Y = []
Z = []

for index in range(1, len(oat.X), step_size):
    X.append(oat.X[index])
    Y.append(oat.Y[index])
    Z.append(oat.Z[index])


'''fig = go.Figure(data=[go.Scatter3d(x=oat.X, y=oat.Y, z=oat.Z,
                                   mode='markers')])'''

#print(oat.__earth_3d())

fig = go.Figure(
    data=[go.Scatter3d(x=X, y=Y, z=Z,
                     mode="lines",
                     line=dict(width=2, color="blue")),
          go.Scatter3d(x=X, y=Y, z=Z,
                     mode="lines",
                     line=dict(width=2, color="blue"))],
    layout=go.Layout(
        #xaxis=dict(range=[-200, 200], autorange=False, zeroline=False),
        #yaxis=dict(range=[-50, 50], autorange=False, zeroline=False),
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