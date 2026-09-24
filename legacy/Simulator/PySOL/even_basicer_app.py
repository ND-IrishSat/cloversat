'''
even_basicer_app.py

'''

import dash
from dash import dcc                           # dash-core-components, i.e, graphs, sliders, dropdowns, etc
from dash import html
from dash.dependencies import Input, Output


import pandas as pd
import plotly.express as px
import geopandas as gpd                         # pandas library to work with geospatial data
import shapely.geometry
import numpy as np
from jupyter_dash import JupyterDash

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import csv
import h5py                                     # Hierarchical Data Format version 5 files are used for managing large datasets,
import astropy.time as astro_time       

import time

from wmm import WMM                             # World Magnetic Model (WMM), a mathematical representation of Earth's magnetic field

# Assigns a low-resolution dataset of country boundaries to 'countries'
countries = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))

# Styling sheets via CodePen
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# Creates a new dash application
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

'''
    Creates a DataFrame
    - assume you have a "long-form" data frame
    - see https://plotly.com/python/px-arguments/ for more options
''' 
df_bar = pd.DataFrame({
    "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
    "Amount": [4, 1, 2, 2, 4, 5],
    "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
})

class OAT:
    '''
        Uses h5py and astropy.time
        Class to initialize and handle the data stored in HDF5 file
    '''
    def __init__(self,fn, path = 'save_sim/', output_path = 'outputs/'):
        '''
            Initialize OAT simulation by loading data from an HDF5 file.
        '''

        print('Initializing OAT simulation..')

        # Set the font globally
        #plt.rcParams.update({'font.family':'sans-serif'})

        # Assigns to where the simulation and output directory
        self.sim_path = path
        self.out_path = output_path

        # Open the HDF5 file
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




fig = make_subplots(rows=2, cols=2,
                    specs=[[{"secondary_y": True}, {"secondary_y": True}],
                           [{"secondary_y": True}, {"secondary_y": True}]])


# Top left
fig.add_trace(
    go.Scatter(x=[1, 2, 3], y=[2, 52, 62], name="yaxis data"),
    row=1, col=1, secondary_y=False)

fig.add_trace(
    go.Scatter(x=[1, 2, 3], y=[40, 50, 60], name="yaxis2 data"),
    row=1, col=1, secondary_y=True,
)

# Top right
fig.add_trace(
    go.Scatter(x=[1, 2, 3], y=[2, 52, 62], name="yaxis3 data"),
    row=1, col=2, secondary_y=False,
)

fig.add_trace(
    go.Scatter(x=[1, 2, 3], y=[40, 50, 60], name="yaxis4 data"),
    row=1, col=2, secondary_y=True,
)

# Bottom left
fig.add_trace(
    go.Scatter(x=[1, 2, 3], y=[2, 52, 62], name="yaxis5 data"),
    row=2, col=1, secondary_y=False,
)

fig.add_trace(
    go.Scatter(x=[1, 2, 3], y=[40, 50, 60], name="yaxis6 data"),
    row=2, col=1, secondary_y=True,
)

# Bottom right
fig.add_trace(
    go.Scatter(x=[1, 2, 3], y=[2, 52, 62], name="yaxis7 data"),
    row=2, col=2, secondary_y=False,
)

fig.add_trace(
    go.Scatter(x=[1, 2, 3], y=[40, 50, 60], name="yaxis8 data"),
    row=2, col=2, secondary_y=True,
)

#fig.show()





















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






fig3 = px.bar(df_bar, x="Fruit", y="Amount", color="City", barmode="group")



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

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)