# -*- coding: utf-8 -*-

# Sleep period list generation function - 21/01/2019
# Julius Andretti
# References:

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go

from scipy.stats import mode
from datetime import datetime

def datetime_diff(stamps):
    datetime_diff = np.array([(stamps[i]-stamps[i-1]).total_seconds() for i in range(1,len(stamps))])
    datetime_diff = np.insert(datetime_diff,0,[0])
    return datetime_diff

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
    return datetime.fromtimestamp(timestamp)

def nights_df(stamps,states,wake_thresh=60,sleep_thresh=120,nap_thresh=20,verbose=False,search_gap=True):
    # inputData is the input array
    # wake_thresh defines for how long one must be awake between sleep periods to diferentiate thems

    if wake_thresh < 1:
        raise Exception('All numerical parameters must be greater than or equal to 1!')

    n = len(states) # Number of points
    
    if not isinstance(stamps[0],datetime):
        if isinstance(stamps[0],str):
            stamps = pd.to_datetime(stamps)
        else:
            stamps = np.array([to_datetime(stamp) for stamp in stamps])

    if search_gap:
        dt_diff = datetime_diff(stamps)
        duration = mode(dt_diff).mode[0]
        time_gaps = np.where(dt_diff > 10*duration,1,0).nonzero()[0]
    else:
        time_gaps = []

    if verbose:
        print("time_gaps",time_gaps)

    boundaries = [[],[]]

    num_gaps = len(time_gaps)
    if num_gaps > 0:
        for i in range(num_gaps):
            if verbose:
                print("i",i)

            if i == 0:
                edges = np.concatenate(([0],np.diff(states[0:time_gaps[i]])))
                edges_index = edges.nonzero()[0]
                transitions = [[k,edges[k]] for k in edges_index]

                if verbose:
                    print("transitions",transitions)

                if len(transitions) > 0:
                    if transitions[0][1] < 0:
                        transitions.insert(0,[0,1])

                    if transitions[-1][1] > 0:
                        transitions.append([time_gaps[i]-1,-1])

                    ini = np.zeros((2,int(0.5*len(transitions))),dtype=int)
                    t = 0
                    for transition in transitions:
                        if transition[1] > 0:
                            ini[0,t] = transition[0]

                        elif transition[1] < 0:
                            ini[1,t] = transition[0]
                            t += 1

                    ini = np.transpose(ini)
                    num_ini = len(ini)
                    k = 1
                    while k < num_ini:
                        if verbose:
                            print("ini",ini)

                        if ini[k][0]-ini[k-1][1] < wake_thresh:
                            ini[k-1][1] = ini[k][1]
                            ini = np.delete(ini,k,0)
                            num_ini = len(ini)

                        else:
                            k += 1

                    for bound in ini:
                        bound_len = bound[1]-bound[0]
                        if bound_len >= sleep_thresh:
                            boundaries[0].append(bound[0])
                            boundaries[1].append(bound[1])
                        else:
                            if np.all(states[bound[0]:bound[1]] == 7):
                                if bound_len >= nap_thresh:
                                    boundaries[0].append(bound[0])
                                    boundaries[1].append(bound[1])
                
                else:
                    boundaries[0].append(0)
                    boundaries[1].append(time_gaps[i]-1)

                if verbose:
                    print("boundaries",np.transpose(boundaries))

            else:
                edges = np.concatenate(([0],np.diff(states[time_gaps[i-1]:time_gaps[i]])))
                edges_index = edges.nonzero()[0]
                transitions = [[k+time_gaps[i-1],edges[k]] for k in edges_index] 
                if verbose:
                    print("transitions",transitions)

                if len(transitions) > 0:
                    if transitions[0][1] < 0:
                        transitions.insert(0,[time_gaps[i-1],1])

                    if transitions[-1][1] > 0:
                        transitions.append([time_gaps[i]-1,-1])

                    ini = np.zeros((2,int(0.5*len(transitions))),dtype=int)
                    t = 0
                    for transition in transitions:
                        if transition[1] > 0:
                            ini[0,t] = transition[0]

                        elif transition[1] < 0:
                            ini[1,t] = transition[0]
                            t += 1

                    ini = np.transpose(ini)
                    num_ini = len(ini)
                    k = 1
                    while k < num_ini:
                        if verbose:
                            print("ini",ini)

                        if ini[k][0]-ini[k-1][1] < wake_thresh:
                            ini[k-1][1] = ini[k][1]
                            ini = np.delete(ini,k,0)
                            num_ini = len(ini)

                        else:
                            k += 1

                    for bound in ini:
                        bound_len = bound[1]-bound[0]
                        if bound_len >= sleep_thresh:
                            boundaries[0].append(bound[0])
                            boundaries[1].append(bound[1])
                        else:
                            if np.all(states[bound[0]:bound[1]] == 7):
                                if bound_len >= nap_thresh:
                                    boundaries[0].append(bound[0])
                                    boundaries[1].append(bound[1])
                
                else:
                    boundaries[0].append(time_gaps[i-1])
                    boundaries[1].append(time_gaps[i]-1)

                if verbose:
                    print("boundaries",np.transpose(boundaries))

            if i == num_gaps-1:
                edges = np.concatenate(([0],np.diff(states[time_gaps[i]:n])))
                edges_index = edges.nonzero()[0]
                transitions = [[k+time_gaps[i],edges[k]] for k in edges_index]
                if verbose:
                    print("transitions",transitions)

                if len(transitions) > 0:
                    if transitions[0][1] < 0:
                        transitions.insert(0,[time_gaps[i],1])

                    if transitions[-1][1] > 0:
                        transitions.append([n-1,-1])

                    ini = np.zeros((2,int(0.5*len(transitions))),dtype=int)
                    t = 0
                    for transition in transitions:
                        if transition[1] > 0:
                            ini[0,t] = transition[0]

                        elif transition[1] < 0:
                            ini[1,t] = transition[0]
                            t += 1

                    ini = np.transpose(ini)
                    num_ini = len(ini)
                    k = 1
                    while k < num_ini:
                        if verbose:
                            print("ini",ini)

                        if ini[k][0]-ini[k-1][1] < wake_thresh:
                            ini[k-1][1] = ini[k][1]
                            ini = np.delete(ini,k,0)
                            num_ini = len(ini)

                        else:
                            k += 1

                    for bound in ini:
                        bound_len = bound[1]-bound[0]
                        if bound_len >= sleep_thresh:
                            boundaries[0].append(bound[0])
                            boundaries[1].append(bound[1])
                        else:
                            if np.all(states[bound[0]:bound[1]] == 7):
                                if bound_len >= nap_thresh:
                                    boundaries[0].append(bound[0])
                                    boundaries[1].append(bound[1])
                
                else:
                    boundaries[0].append(time_gaps[i])
                    boundaries[1].append(n-1)

                if verbose:
                    print("boundaries",np.transpose(boundaries))

    else:
        edges = np.concatenate(([0],np.diff(states)))
        edges_index = edges.nonzero()[0]
        transitions = [[k,edges[k]] for k in edges_index]

        if verbose:
            print("transitions",transitions)

        if len(transitions) > 0:
            if transitions[0][1] < 0:
                transitions.insert(0,[0,1])

            if transitions[-1][1] > 0:
                transitions.append([n-1,-1])


            ini = np.zeros((2,int(0.5*len(transitions))),dtype=int)
            t = 0
            for transition in transitions:
                if transition[1] > 0:
                    ini[0,t] = transition[0]

                elif transition[1] < 0:
                    ini[1,t] = transition[0]
                    t += 1

            ini = np.transpose(ini)
            # print(ini)

            num_ini = len(ini)
            k = 1
            while k < num_ini:
                if verbose:
                    print("ini",ini)

                if ini[k][0]-ini[k-1][1] < wake_thresh:
                    ini[k-1][1] = ini[k][1]
                    ini = np.delete(ini,k,0)
                    num_ini = len(ini)

                else:
                    k += 1

            for bound in ini:
                bound_len = bound[1]-bound[0]
                if bound_len >= sleep_thresh:
                    boundaries[0].append(bound[0])
                    boundaries[1].append(bound[1])
                else:
                    if np.all(states[bound[0]:bound[1]] == 7):
                        if bound_len >= nap_thresh:
                            boundaries[0].append(bound[0])
                            boundaries[1].append(bound[1])
        
        else:
            boundaries[0].append(0)
            boundaries[1].append(n-1)

        if verbose:
            print("boundaries",np.transpose(boundaries))

    boundaries = np.transpose(boundaries)
    num_ini = len(boundaries)
    k = 1
    while k < num_ini:
        if verbose:
            print("boundaries",boundaries)

        if boundaries[k][0]-boundaries[k-1][1] < wake_thresh:
            boundaries[k-1][1] = boundaries[k][1]
            boundaries = np.delete(boundaries,k,0)
            num_ini = len(boundaries)

        else:
            k += 1

    # layout = go.Layout(title="",xaxis=dict(title="Date time"), yaxis=dict(title="Inputs"), showlegend=False)
    # layout.update(yaxis2=dict(overlaying='y',side='right'), showlegend=True)

    # fig = go.Figure(data=[
    #     go.Scatter(x=stamps.astype(str),y=states,  yaxis='y2',name='act'),
    #     go.Scatter(x=stamps.astype(str),y=np.concatenate(([0],np.diff(states))),  yaxis='y2',name='diff'),
    #     go.Scatter(x=stamps.astype(str),y=dt_diff, name='dt_diff'),
    #     # go.Scatter(x=stamps.astype(str),y=out, yaxis='y2', name='out'),
    # ], layout=layout)

    # fig.show()

    if verbose:
        for item in boundaries:
            print(stamps[item[0]],stamps[item[1]])

    df = pd.DataFrame(boundaries,columns=["bt","gt"])
    df["nap"] = False

    t = 0
    for bound in boundaries:
        if np.all(states[bound[0]:bound[1]] == 7):
            df.at[t,"nap"] = True
        t += 1

    return df