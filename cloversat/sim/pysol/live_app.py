#https://dash.plotly.com/live-updates

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



import datetime

import plotly
from dash.dependencies import Input, Output

# pip install pyorbital
from pyorbital.orbital import Orbital
#satellite = Orbital('TERRA')

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


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

def get_time(time):
    if time < 0:
        time = 0
    return oat.times_utc[time]

def get_lonlat(time):
    if time < 0:
        time = 0
    return oat.LALN[time][1], oat.LALN[time][0]

def get_B(time):
    if time < 0:
        time = 0
    return oat.B[time], oat.Bx[time], oat.By[time], oat.Bz[time]

def get_XYZ(time):
    if time < 0:
        time = 0
    return oat.X[time], oat.Y[time], oat.Z[time]



app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div(
    html.Div([
        html.H4('IrishSat Demo Dashboard'),
        html.Div(id='live-update-text'),
        dcc.Graph(id='live-update-graph'),
        dcc.Interval(
            id='interval-component',
            interval=1*1000, # in milliseconds
            n_intervals=0
        )
    ])
)





@app.callback(Output('live-update-text', 'children'),
              Input('interval-component', 'n_intervals'))
def update_metrics(n):
    #print(n*50)
    #lon, lat = get_lonlat(n*50)
    #curr_time = oat.times_utc[n*50]
    fast_n = n*40+2000

    lon, lat = get_lonlat(fast_n)
    curr_time = get_time(fast_n)
    style = {'padding': '5px', 'fontSize': '16px'}
    return [
        html.Span('Longitude: {0:.2f}'.format(lon), style=style),
        html.Span('Latitude: {0:.2f}\n'.format(lat), style=style),
        html.Span('Time: {} '.format(str(curr_time)), style=style),
        html.Span('f: {:3.2f}° '.format(oat.OE_[fast_n, 0])),
        html.Span('a: {:5.2f} [km] '.format(oat.OE_[fast_n, 1])),
        html.Span('e: {:3.5f}\n'.format(oat.OE_[fast_n, 2])),
        html.Span('i: {:3.2f}° '.format(oat.OE_[fast_n, 3])),
        html.Span('Ω: {:3.2f}° '.format(oat.OE_[fast_n, 4])),
        html.Span('h: {:3.2f} [km]'.format(oat.H[fast_n])),
    ]
    '''lon, lat, alt = satellite.get_lonlatalt(datetime.datetime.now())
    style = {'padding': '5px', 'fontSize': '16px'}
    return [
        html.Span('Longitude: {0:.2f}'.format(lon), style=style),
        html.Span('Latitude: {0:.2f}'.format(lat), style=style),
        html.Span('Altitude: {0:0.2f}'.format(alt), style=style)
    ]'''


# Multiple components can update everytime interval gets fired.
@app.callback(Output('live-update-graph', 'figure'),
              Input('interval-component', 'n_intervals'))
def update_graph_live(n):
    '''satellite = Orbital('TERRA')
    data = {
        'time': [],
        'Latitude': [],
        'Longitude': [],
        'Altitude': []
    }

    # Collect some data
    for i in range(180):
        time = datetime.datetime.now() - datetime.timedelta(seconds=i*20)
        lon, lat, alt = satellite.get_lonlatalt(
            time
        )
        data['Longitude'].append(lon)
        data['Latitude'].append(lat)
        data['Altitude'].append(alt)
        data['time'].append(time)

    # Create the graph with subplots
    fig = plotly.tools.make_subplots(rows=2, cols=1, vertical_spacing=0.2)
    fig['layout']['margin'] = {
        'l': 30, 'r': 10, 'b': 30, 't': 10
    }
    fig['layout']['legend'] = {'x': 0, 'y': 1, 'xanchor': 'left'}

    fig.append_trace({
        'x': data['time'],
        'y': data['Altitude'],
        'name': 'Altitude',
        'mode': 'lines+markers',
        'type': 'scatter'
    }, 1, 1)
    fig.append_trace({
        'x': data['Longitude'],
        'y': data['Latitude'],
        'text': data['time'],
        'name': 'Longitude vs Latitude',
        'mode': 'lines+markers',
        'type': 'scatter'
    }, 2, 1)'''


    data = {
        'time': [],
        'Latitude': [],
        'Longitude': [],
        '|B|': [],
        'Bx': [],
        'By': [],
        'Bz': [],
        'X': [],
        'Y': [],
        'Z': []
    }

    fast_n = n*40+2000

    for i in range(180):
        time = get_time(fast_n) - datetime.timedelta(seconds=i*20)
        lon, lat = get_lonlat(fast_n-i*20)
        B, Bx, By, Bz = get_B(fast_n-i*20)
        X, Y, Z = get_XYZ(fast_n-i*20)
        data['Longitude'].append(lon)
        data['Latitude'].append(lat)
        data['|B|'].append(B)
        data['Bx'].append(Bx)
        data['By'].append(By)
        data['Bz'].append(Bz)
        data['X'].append(X)
        data['Y'].append(Y)
        data['Z'].append(Z)
        data['time'].append(time)

    #B, Bx, By, Bz = get_B(n*50)
    #lons, lats = get_lonlat(n*50)

    fig = plotly.tools.make_subplots(
        rows=3, cols=1,
        specs=[[{"type": "scattergeo"}],
            [{"type": "scatter"}],
            [{"type": "scatter3d"}]]
    )

    fig.add_trace(
        go.Scattergeo(lon=data['Longitude'], lat=data['Latitude'],
                     mode="lines",
                     line=dict(width=2, color="red")),
        row=1, col=1
    )

    # fix this
    fig.add_trace(
        go.Scatter(name='|B|', x=data['time'], y=data['|B|']),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(name='Bx', x=data['time'], y=data['Bx']),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(name='By', x=data['time'], y=data['By']),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(name='Bz', x=data['time'], y=data['Bz']),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter3d(x=data['X'], y=data['Y'], z=data['Z'], mode="lines"),
        row=3, col=1
    )

    fig.update(layout_yaxis_range = [-60,60])

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
