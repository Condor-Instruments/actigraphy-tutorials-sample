import os, sys, inspect

import time
import pandas as pd
import numpy as np
import plotly.graph_objs as go

import plotly.io as pio
pio.renderers.default='browser'

from plotly.subplots import make_subplots
from scipy.stats import mode
from datetime import datetime, timedelta

from cspd_functions import *


def to_datetime(date):
    """
    Converts a numpy datetime64 object to a python datetime object 
    Input:
      date - a np.datetime64 object
    Output:
      DATE - a python datetime object
    """
    timestamp = ((date - np.datetime64('1970-01-01T00:00:00'))
                 / np.timedelta64(1, 's'))
    return datetime.utcfromtimestamp(timestamp)

    
color_list = [
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf'   # blue-teal
    ]

def actigraphy_split_by_day(df, start_hour = 0, dt=None):
    ldays = []
    ldays_ref = []

    if dt is None:
        dt = df.index

    else:
        dt = df[dt]

    n = len(dt)

    if not isinstance(dt[0],datetime):
        if isinstance(dt[0],str):
            dt = pd.to_datetime(dt,dayfirst=True)
        
    dt = np.array([to_datetime(stamp) for stamp in dt])
    # else:
    #     dt = dt.values

    df.index = dt
    
    #First day is the start of the day of the first Epoch
    sdate = pd.Timestamp(year = dt[0].year, month = dt[0].month, day = dt[0].day)
    if(dt[0].hour <= start_hour):
        sdate = sdate - pd.Timedelta(hours = start_hour) 
    else:
        sdate = sdate + pd.Timedelta(hours = start_hour)         
    
    while sdate < dt[n-1]:
        day = np.logical_and(dt >= sdate, dt < sdate + pd.Timedelta(hours = 24))
        if len(df[day]) > 0:
            ldays.append(df[day])
            ldays_ref.append(pd.Timestamp(year = sdate.year, month = sdate.month, day = sdate.day))
        sdate += pd.Timedelta(hours = 24)
    
    return ldays, ldays_ref

def actigraphy_single_plot_actogram(df, cols, secondary_y, start_hour, title, dt = None):
    ldays = []
    ldays_ref = []
    
    #First day is the start of the day of the first Epoch
    ldays, ldays_ref = actigraphy_split_by_day(df, start_hour, dt)

    specs = [[{"secondary_y": True}]] * len(ldays)
    fig = make_subplots(len(ldays), 1, subplot_titles=[title],
                        specs = specs)
    
    for i in range(len(ldays)):
        j = 0
        showlegend = False
        if(i == 1):
            showlegend = True

        for cn in cols: 
            fig.add_trace(go.Scatter(x=ldays[i].index,
                                     y=ldays[i][cn].to_numpy(),
                                     opacity=0.8, 
                                     fillcolor= 'rgb(0,160,255)',
                                     line=dict(color=color_list[j%len(color_list)]),
                                     legendgroup=cn,
                                     showlegend=showlegend,
                                     name = cn,
                                     ),
                          i+1, 1,
                          secondary_y=secondary_y[j]
                          )
            j+=1
        fig.update_xaxes(range=[ldays_ref[i] + pd.Timedelta(hours = 12), ldays_ref[i] + pd.Timedelta(hours = 36)], row=i+1, col=1)
    fig.update_layout(height=len(ldays)*400)

    return fig    