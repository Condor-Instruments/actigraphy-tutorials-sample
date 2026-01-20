# -*- coding: utf-8 -*-

# Sleep-analysis-related functions - 02/2021
# Julius Andretti
# References:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, auc
import jax.numpy as jnp
from datetime import date,datetime,timedelta,time
from scipy.signal import find_peaks as peak


def actigraphy_split_by_day(signal,timestamps,start_hour=12):
    splits = []
    dates = []

    n = len(timestamps)

    # if not isinstance(timestamps[0],datetime):
    #     if isinstance(timestamps[0],str):
    #         timestamps = pd.to_datetime(timestamps,dayfirst=True)
        
    # timestamps = np.array([to_datetime(stamp) for stamp in timestamps])
    # else:
    #     timestamps = timestamps.values

    signal = pd.DataFrame(signal,columns=["signal"])
    # print(signal)
    signal.index = timestamps
    
    #First day is the start of the day of the first Epoch
    first_date = pd.Timestamp(year = timestamps[0].year, month = timestamps[0].month, day = timestamps[0].day)
    if(timestamps[0].hour <= start_hour):
        first_date = first_date - pd.Timedelta(hours = start_hour) 
    else:
        first_date = first_date + pd.Timedelta(hours = start_hour)         
    
    current_date = first_date
    while current_date < timestamps[n-1]:
        day = np.logical_and(timestamps >= current_date, timestamps < current_date + pd.Timedelta(hours=24))
        if len(signal[day]) > 0:
            splits.append(np.transpose(signal[day].values)[0])
            dates.append(pd.Timestamp(year = current_date.year, month = current_date.month, day = current_date.day))
        current_date += pd.Timedelta(hours = 24)

    days = len(dates)
    if days > 2:
        typical_length = len(splits[1])
        if typical_length >= 2*len(splits[0]):        
            first_split = [np.hstack((splits[0],splits[1]))]
            splits = first_split + splits[2:days]
            dates = dates[1:days]
            days -= 1

        if typical_length >= 2*len(splits[days-1]):
            last_split = [np.hstack((splits[days-2],splits[days-1]))]
            splits = splits[0:days-2] + last_split
            dates = dates[0:days-1]
            days -= 1

    elif days == 2:
        if len(splits[0]) >= 2*len(splits[1]):
            dates = [dates[0]]
            splits = [np.hstack((splits[0],splits[1]))]
        
        elif len(splits[1]) >= 2*len(splits[0]):        
            splits = [np.hstack((splits[0],splits[1]))]
            dates = [dates[1]]

    return splits, dates



def diff5(signal,delta=1):  # Computa a estimativa da derivada utilizando a fórmula da "diferença de 5 pontos"
    signal = np.array(signal)

    n = len(signal)

    diff5 = [(1/(12*delta))*(signal[i-2]-8*signal[i-1]+8*signal[i+1]-signal[i+2]) for i in range(2,n-2)]
    diff5 = np.insert(diff5.copy(),0,[(1/(12*delta))*(-25*signal[0]+48*signal[1]-36*signal[2]+16*signal[3]-3*signal[4]),(1/(12*delta))*(-25*signal[1]+48*signal[2]-36*signal[3]+16*signal[4]-3*signal[5])])
    diff5 = np.append(diff5,[(1/(12*delta))*(25*signal[n-2]-48*signal[n-3]+36*signal[n-4]-16*signal[n-5]+3*signal[n-6]),(1/(12*delta))*(25*signal[n-1]-48*signal[n-2]+36*signal[n-3]-16*signal[n-4]+3*signal[n-5])])

    return diff5


def pad_signal(signal,pad_size,pad_with=[]):
    """Pads a signal in the beginning and the end with given values
    or the respective border values

    Parameters
    ----------
    signal : np.array
        Time series to be padded
    pad_size : int
        Number of epochs in the pad

    Returns
    -------
    padded_signal : np.array
        Time series with padding
    """

    signal_length = len(signal)

    pad_value_count = len(pad_with)
    if pad_value_count==0:
        beginning_pad = signal[0]
        ending_pad = signal[signal_length-1]
    elif pad_value_count==1:
        beginning_pad = pad_with[0]
        ending_pad = beginning_pad
    else:
        beginning_pad = pad_with[0]
        ending_pad = pad_with[1]

    beginning_pad = beginning_pad*np.ones(pad_size)
    ending_pad = ending_pad*np.ones(pad_size)

    padded_signal = np.insert(signal.copy(),0,beginning_pad)
    padded_signal = np.append(padded_signal,ending_pad)

    return padded_signal


def extract_features(signal,half_window_length,features,column_prefix):
    n = len(signal)
    max_half_window_length = half_window_length[-1]

    loop_start = max_half_window_length
    loop_end = n+max_half_window_length

    first_epoch = signal[0]
    last_epoch = signal[n-1]

    beginning_pad = first_epoch*np.ones(max_half_window_length)
    ending_pad = last_epoch*np.ones(max_half_window_length)

    padded_signal = np.insert(signal.copy(),0,beginning_pad)
    padded_signal = np.append(padded_signal,ending_pad)

    data = pd.DataFrame([])

    if "signal" in features:
        data[column_prefix+"signal"] = signal.copy()

    if "full_zp" in features:
       data[column_prefix+"full_zp"] = zero_prop(signal)

    for half_window in half_window_length:
        rolled = rolling_window(padded_signal[loop_start-half_window:loop_end+half_window], 2*half_window+1)

        if "mean" in features:
            data[column_prefix+'mean_w='+str(2*half_window+1)] = np.mean(rolled, axis=-1)[0:n]
            
        if "median" in features:
            data[column_prefix+'median_w='+str(2*half_window+1)] = np.median(rolled, axis=-1)[0:n]
            
        if "std" in features:
            data[column_prefix+'std_w='+str(2*half_window+1)] = np.std(rolled, axis=-1)[0:n]
            
        if "var" in features:
            data[column_prefix+'var_w='+str(2*half_window+1)] = np.var(rolled, axis=-1)[0:n]

        if "max" in features:
            data[column_prefix+'max_w='+str(2*half_window+1)] = np.max(rolled, axis=-1)[0:n]

        if "zp" in features:
            data[column_prefix+'zp_w='+str(2*half_window+1)] = zp_axis(rolled)[0:n]

    return data

def datetime_diff(stamps):
    datetime_diff = np.array([(stamps[i]-stamps[i-1]).total_seconds() for i in range(1,len(stamps))])
    datetime_diff = np.insert(datetime_diff,0,[0])
    return datetime_diff

def datetime_distance(stamps,datetime_input):
    datetime_distance = np.array([(stamps[i]-datetime_input).total_seconds() for i in range(len(stamps))])
    return datetime_distance

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

# Scales input to be in the [-1,1] interval 
def scale_by_max(signal,scale=None):
    # signal is the input data

    if len(signal) > 0:
        if scale is None:   # Data is divided by its largest value
            factor = 1.0/np.max(np.absolute(signal))
        else:   # Data is the divided by given scalar
            factor = 1.0/scale
        return factor*np.array(np.absolute(signal))
    else:
        return signal


def norm_01(signal,normalize_like=[]):
    if len(signal) > 0:
        if len(normalize_like) > 0:
            mini = np.min(normalize_like)
            span = np.max(normalize_like)-mini
        else:
            mini = np.min(signal)
            span = np.max(signal)-mini
        
        scale = 1.0/span
        return scale*signal-mini*scale
    else:
        return signal       


def zero_prop(a):
    if len(a) > 0:
        return  np.count_nonzero(a==0)/len(a)
    else:
        return 0.0


def iqr_computation(x):
    iqr = 0.0
    if len(x) > 0:
        q75, q25 = np.percentile(x, [75 ,25])
        iqr = q75 - q25        

    return iqr


def segmentation(data,nbins=10):
    total = np.sum(data)
    length = len(data)

    bins = np.linspace(0,length,num=nbins+1,endpoint=True,dtype=int)
    segmentation = np.zeros(nbins)

    for b in range(1,nbins+1):
        start = bins[b-1]
        end = bins[b]
        
        if total > 0:
            segmentation[b-1] = np.sum(data[start:end])/total
        else:
            segmentation[b-1] = 0

    return segmentation


def zero_sequences(a,minimum_length=1,ending_at_n=False):
    n = len(a)
    
    if n > 1:
        edges = np.concatenate(([0],np.diff(a)))
        edges_index = edges.nonzero()[0]
        transitions = [[k,edges[k]] for k in edges_index]

        if len(transitions) > 0:
            if transitions[0][1] > 0:
                transitions.insert(0,[0,-1])

            if transitions[-1][1] < 0:
                if ending_at_n:
                    transitions.append([n,1])
                else:
                    transitions.append([n-1,1])
        else:
            if np.sum(a) == 0:
                if ending_at_n:
                    transitions = [[0,-1],[n,1]]
                else:
                    transitions = [[0,-1],[n-1,1]]

        seqs = [[],[]]
        for transition in transitions:
            if transition[1] < 0:
                seqs[0].append(transition[0])

            elif transition[1] > 0:
                seqs[1].append(transition[0])

        
        # print(seqs)
        # print(len(seqs[0]))
        # print(len(seqs[1]))

        seqs = np.transpose(seqs)
        seq_length = np.array([o[1]-o[0] for o in seqs])
        seqs = seqs[np.where(seq_length >= minimum_length)]

        return seqs
    else:
        return np.array([])

def get_peak(signal,center,valley=False,peak_hws=10):
    if (center-peak_hws) >= 0:
        if (center+peak_hws+1) <= len(signal):
            peak_search = signal[center-peak_hws:center+peak_hws+1]
        else:
            peak_search = signal[center-peak_hws:len(signal)]
    else:
        peak_search = signal[0:center+peak_hws+1]

    # print(signal, len(signal))
    # print(center)
    # print(peak_search)

    if valley:
        peak = np.argmin(peak_search)
    else:
        peak = np.argmax(peak_search)
    
    return peak_search[peak]


def rolling_window(a, window, step=1):
    a = np.array(a)
    total_len_x = a.shape[-1]

    if total_len_x >= window:
        new_shape_row = (total_len_x - window)//step + 1
        new_shape_col = window
        new_shape = (new_shape_row, new_shape_col)  

        if len(a.shape) > 1:
            n_bytes = a.strides[-1]
            stride_steps_row = n_bytes * step
            stride_steps_col = n_bytes
            stride_steps = (stride_steps_row, stride_steps_col)
        else:
            stride_steps = a.strides + (a.strides[-1],)

        return np.lib.stride_tricks.as_strided(a, shape=new_shape, strides=stride_steps)
    else:
        return np.nan*np.ones(window)

def zp_axis(rolled):
    return np.apply_along_axis(zero_prop,-1,rolled)

def iqr_axis(rolled):
    return np.apply_along_axis(iqr_computation,-1,rolled)

def median_filter(signal,hws,padding=None,center=True,forward=True):
    # signal is the input data
    # hws is the size of the half-window used
    # choice is an integer used to choose between median and max filtering
    # padding is a list containing padding information
    signal = np.array(signal)
    n = len(signal)

    if n > hws:
        if padding == None:
            pad = signal[0]*np.ones(2*hws)
            filt = np.insert(signal.copy(),0,pad)
            pad = signal[n-1]*np.ones(2*hws)
            filt = np.append(filt,pad)
            rolled = rolling_window(filt, 2*hws+1)

        elif len(padding) == 2:
            pad = padding[0]*np.ones(2*hws)
            filt = np.insert(signal.copy(),0,pad)
            pad = padding[1]*np.ones(2*hws)
            filt = np.append(filt,pad)
            rolled = rolling_window(filt, 2*hws+1)

        elif padding == "padded":
            n -= 4*hws
            rolled = rolling_window(signal, 2*hws+1)
        
        # filt = np.median(rolled, axis=-1)[0:n]

        if center:
            filt = np.median(rolled, axis=-1)[hws:n+hws]
        else:
            if forward:
                filt = np.median(rolled, axis=-1)[2*hws:n+2*hws]
            else:
                filt = np.median(rolled, axis=-1)[0:n]

        return filt
    else:
        return signal
    

def quantile_filter(signal,hws,quantile=0.6,method='inverted_cdf',padding=None,center=True,forward=True):
    # signal is the input data
    # hws is the size of the half-window used
    # choice is an integer used to choose between median and max filtering
    # padding is a list containing padding information

    signal = np.array(signal)
    n = len(signal)

    if n > hws:
        if padding == None:
            pad = signal[0]*np.ones(2*hws)
            filt = np.insert(signal.copy(),0,pad)
            pad = signal[n-1]*np.ones(2*hws)
            filt = np.append(filt,pad)
            rolled = rolling_window(filt, 2*hws+1)

        elif len(padding) == 2:
            pad = padding[0]*np.ones(2*hws)
            filt = np.insert(signal.copy(),0,pad)
            pad = padding[1]*np.ones(2*hws)
            filt = np.append(filt,pad)
            rolled = rolling_window(filt, 2*hws+1)

        elif padding == "padded":
            n -= 4*hws
            rolled = rolling_window(signal, 2*hws+1)
        
        # filt = np.median(rolled, axis=-1)[0:n]

        if center:
            filt = np.quantile(rolled,quantile,method=method,axis=-1)[hws:n+hws]
        else:
            if forward:
                filt = np.quantile(rolled,quantile,method=method,axis=-1)[2*hws:n+2*hws]
            else:
                filt = np.quantile(rolled,quantile,method=method,axis=-1)[0:n]

        return filt
    else:
        return signal


def mean_filter(signal,hws,padding=None):
    # signal is the input data
    # hws is the size of the half-window used
    # choice is an integer used to choose between median and max filtering
    # padding is a list containing padding information
    signal = np.array(signal)
    n = len(signal)

    if n > hws:
        if padding == None:
            pad = signal[0]*np.ones(hws)
            filt = np.insert(signal.copy(),0,pad)
            pad = signal[n-1]*np.ones(hws)
            filt = np.append(filt,pad)
            rolled = rolling_window(filt, 2*hws+1)

        elif len(padding) == 2:
            pad = padding[0]*np.ones(hws)
            filt = np.insert(signal.copy(),0,pad)
            pad = padding[1]*np.ones(hws)
            filt = np.append(filt,pad)
            rolled = rolling_window(filt, 2*hws+1)

        elif padding == "padded":
            n -= 2*hws
            rolled = rolling_window(signal, 2*hws+1)

        filt = np.mean(rolled, axis=-1)[0:n]
        return filt

    else:
        return signal

def var_filter(signal,hws,padding=None):
    # signal is the input data
    # hws is the size of the half-window used
    # choice is an integer used to choose between median and max filtering
    # padding is a list containing padding information
    signal = np.array(signal)
    n = len(signal)

    if n > hws:
        if padding == None:
            pad = signal[0]*np.ones(hws)
            filt = np.insert(signal.copy(),0,pad)
            pad = signal[n-1]*np.ones(hws)
            filt = np.append(filt,pad)
            rolled = rolling_window(filt, 2*hws+1)

        elif len(padding) == 2:
            pad = padding[0]*np.ones(hws)
            filt = np.insert(signal.copy(),0,pad)
            pad = padding[1]*np.ones(hws)
            filt = np.append(filt,pad)
            rolled = rolling_window(filt, 2*hws+1)

        elif padding == "padded":
            n -= 2*hws
            rolled = rolling_window(signal, 2*hws+1)
        
        filt = np.var(rolled, axis=-1)[0:n]
        return filt

    else:
        return signal

def iqr_filter(signal,hws,padding=None):
    # signal is the input data
    # hws is the size of the half-window used
    # choice is an integer used to choose between median and max filtering
    # padding is a list containing padding information
    signal = np.array(signal)
    n = len(signal)

    if n > hws:
        if padding == None:
            pad = signal[0]*np.ones(hws)
            filt = np.insert(signal.copy(),0,pad)
            pad = signal[n-1]*np.ones(hws)
            filt = np.append(filt,pad)
            rolled = rolling_window(filt, 2*hws+1)

        elif len(padding) == 2:
            pad = padding[0]*np.ones(hws)
            filt = np.insert(signal.copy(),0,pad)
            pad = padding[1]*np.ones(hws)
            filt = np.append(filt,pad)
            rolled = rolling_window(filt, 2*hws+1)

        elif padding == "padded":
            n -= 2*hws
            rolled = rolling_window(signal, 2*hws+1)
        
        filt = iqr_axis(rolled)[0:n]
        return filt

    else:
        return signal



def zero_prop_filter(signal,hws,padding=[1,1]):
    # signal is the input data
    # hws is the size of the half-window used
    # choice is an integer used to choose between median and max filtering
    # padding is a list containing padding information

    n = len(signal)

    if n > hws:
        if len(padding) == 2:
            pad = padding[0]*np.ones(hws)
            filt = np.insert(signal.copy(),0,pad)
            pad = padding[1]*np.ones(hws)
            filt = np.append(filt,pad)
            rolled = rolling_window(filt, 2*hws+1)
            filt = zp_axis(rolled)[0:n]

        elif padding == "padded":
            n -= 2*hws
            rolled = rolling_window(signal, 2*hws+1)
            filt = zp_axis(rolled)[0:n]

        return filt
    else:
        return signal
    

def fourier_filter(signal, k, lowpass): # Funçao que realiza a filtragem passa-altas ou passa-baixas do sinal de
                                        # acordo com o valor do parâmetro lowpass. k é o parâmetro de corte. 
    direct = np.fft.rfft(signal)
    n = len(direct)

    filtro = np.zeros(n)
    filtro[0] = direct[0]

    if lowpass:
        for i in range(1,k+1):
            filtro[i] = direct[i]
        for i in range(n-(k+1),n): # Necessário para considerar a simetria dos coeficientes #
            filtro[i] = direct[i]
    else:
        for i in range(k,n-k-1):    
            filtro[i] = direct[i]
            
    return np.fft.irfft(filtro,len(signal))


def rolling_sum(signal,hws,padding=None,center=True,forward=True):
    # signal is the input data
    # hws is the size of the half-window used
    # choice is an integer used to choose between median and max filtering
    # padding is a list containing padding information
    signal = np.array(signal)
    n = len(signal)

    if n > hws:
        if padding == None:
            pad = signal[0]*np.ones(2*hws)
            filt = np.insert(signal.copy(),0,pad)
            pad = signal[n-1]*np.ones(2*hws)
            filt = np.append(filt,pad)
            rolled = rolling_window(filt, 2*hws+1)

        elif len(padding) == 2:
            pad = padding[0]*np.ones(2*hws)
            filt = np.insert(signal.copy(),0,pad)
            pad = padding[1]*np.ones(2*hws)
            filt = np.append(filt,pad)
            rolled = rolling_window(filt, 2*hws+1)

        elif padding == "padded":
            n -= 2*hws
            rolled = rolling_window(signal, 2*hws+1)
        
        if center:
            filt = np.sum(rolled, axis=-1)[hws:n+hws]
        else:
            if forward:
                filt = np.sum(rolled, axis=-1)[2*hws:n+2*hws]
            else:
                filt = np.sum(rolled, axis=-1)[0:n]
        

        return filt

    else:
        return signal


def roc_curve(answer,prob,over=True,threshs=100,):
    min_thresh = np.min(prob)
    max_thresh = np.max(prob)
    thresholds = np.linspace(min_thresh,max_thresh,threshs)

    fpr = []
    tpr = []
    for thresh in thresholds:
        if over:
            predicted = np.where(prob >= thresh,1,0)
        else:
            predicted = np.where(prob < thresh,1,0)

        tn, fp, fn, tp = confusion_matrix(answer,predicted).ravel()  # Calculates statistical scores
        fpr.append(fp/(fp+tn))
        tpr.append(tp/(tp+fn))

    return fpr,tpr

def below_prop(a,thresh):
    if len(a) > 0:
        return np.sum(np.where(a <= thresh,1,0))/len(a)
    else:
        return 0.0

# Calculates actigraphic scores
def scores(true, predicted, valid_event_on=0.5, valid_event_off=0.5,states=None,long_event=240,duration=60,ofs_locate=False):
    # true is the desired output
    # predicted is self-explanatory
    # Sequences of event are being analyzed
    # valid_event is the maximal percentage of error needed to classify an event as detected

    true = np.array(true)
    predicted = np.array(predicted)

    accuracy = 0
    sensitivity = 0
    specificity = 0
    precision = 0
    negative_predictive_value = 0

    ravel = confusion_matrix(true,predicted).ravel()  # Calculates statistical scores
    print(ravel)
    if len(ravel) > 1:
        tn, fp, fn, tp = ravel
    
        accuracy = (tp+tn)/(tn+fn+fp+tp)
        
        if tp > 0:
            sensitivity = tp/(tp+fn)
            precision = tp/(tp+fp)

        if tn > 0:
            specificity = tn/(tn+fp)
            negative_predictive_value = tn/(tn+fn)

    else:
        accuracy = 1
        sensitivity = 1
        specificity = 1
        precision = 1
        negative_predictive_value = 1

    n = len(true)
    
    # print(np.sum(true))
    # print(accuracy)
    # print(sensitivity)
    # print(specificity)
    # print(precision)

    if np.sum(true) == n:
        if sensitivity == 1.0:
            specificity = 1.0
            negative_predictive_value = 1.0

    elif np.sum(true) == 0:
        if specificity == 1.0:
            sensitivity = 1.0
            precision = 1.0
           
    long_event = int(round(long_event*60/duration))

    if states is None:
        # We'll calculate the begginings and endings of sequences of 1s to calculate the error inside them
        edges = np.concatenate([[0], np.diff(true)])

        begin = (edges > 0).nonzero()[0]
        end = (edges < 0).nonzero()[0]

        if true[0] == 1:
            begin = np.insert(begin,0,0)

        if true[-1] == 1:
            end = np.append(end,n)

        over_found = 0
        event_count = len(begin)   # Number of events in true
        valid = 0
        detection = 0

        if event_count > 0:
            for i in range(event_count):
                event = predicted[begin[i]:end[i]]

                len_event = end[i]-begin[i]  # Length of event

                error_count = len_event - sum(event)   # Number of erroneous detections

                edges = np.concatenate([[0], np.diff(event)])
                begin_ = (edges < 0).nonzero()[0]
                over_found += len(begin_)

                error = error_count/len_event   # Error percentage

                if error <= valid_event_on:
                    valid += 1

            detection = valid/event_count

        else:
            if sensitivity == 1:
                detection = 1

        detection_on,valid_on,event_count_on = [detection,valid,event_count]


        begin = (edges < 0).nonzero()[0]
        end = (edges > 0).nonzero()[0]

        if true[0] == 0:
            begin = np.insert(begin,0,0)

        if true[-1] == 0:
            end = np.append(end,n)

        event_count = len(begin)   # Number of events in true
        valid = 0
        detection = 0
        if event_count > 0:
            for i in range(event_count):
                if i < len(end):
                    len_event = end[i]-begin[i]
                    error_count = sum(predicted[begin[i]:end[i]])
                else:
                    len_event = n-begin[i]
                    error_count = sum(predicted[begin[i]:n])

                error = error_count/len_event   # Error percentage

                if error <= valid_event_off:
                    valid += 1

            detection = valid/event_count

        else:
            if specificity == 1:
                detection = 1

        detection_off,valid_off,event_count_off = [detection,valid,event_count]

        not_found = event_count_off - valid_off

        return [accuracy, sensitivity, specificity, precision, negative_predictive_value, (detection_on,valid_on,event_count_on), (detection_off,valid_off,event_count_off), over_found]

    else:
        # We'll calculate the begginings and endings of sequences of 1s to calculate the error inside them
        edges = np.concatenate([[0], np.diff(true)])

        begin = (edges > 0).nonzero()[0]
        end = (edges < 0).nonzero()[0]

        if true[0] == 1:
            begin = np.insert(begin,0,0)

        if true[-1] == 1:
            end = np.append(end,n)

        over_found = 0
        over_found_sleep = 0
        event_count = len(begin)   # Number of events in true
        valid = 0
        detection = 0
        of_location = []
        ofs_location = []

        # print("begin",begin)
        # print("end",end)
        # print("event_count",event_count)

        if event_count > 0:
            for i in range(event_count):
                event = predicted[begin[i]:end[i]]

                len_event = end[i]-begin[i]  # Length of event
                error_count = len_event - sum(event)   # Number of erroneous detections

                edges_ = np.concatenate([[0], np.diff(event)])
                begin_ = (edges_ < 0).nonzero()[0]
                end_ = (edges_ > 0).nonzero()[0]

                # if i == 0:
                #     if begin[i] == 0:
                #         if event[0] == 0:
                #             begin_ = np.insert(begin_,0,0)

                # if event[-1] == 0:
                #     end_ = np.append(end_,len_event)
                    
                blen = len(begin_)
                elen = len(end_)

                if blen+elen > 0:
                    # print("begin_",begin_)
                    # print("end_",end_)

                    start = 0
                    ending = blen

                    if blen > elen:
                        end_ = np.append(end_,len_event)
                        ending -= 1
                        
                    elif blen < elen:
                        begin_ = np.insert(begin_,0,0)
                        start += 1
                        ending += 1

                    else:
                        if end_[0] < begin_[0]:
                            end_ = np.append(end_,len_event)
                            begin_ = np.insert(begin_,0,0)
                            start += 1

                    # print("begin_",begin_)
                    # print("end_",end_)
                    # print("start",start,"ending",ending)
                    # input()

                    over_found += (ending - start)

                    begin_ = begin_ + begin[i]
                    end_ = end_ + begin[i]

                    for b in range(start,ending):
                        of_location.append([begin_[b],end_[b]])
                        if zero_prop(states[begin_[b]:end_[b]]) < 0.5:
                            over_found_sleep += 1
                            ofs_location.append([begin_[b],end_[b]])

                error = error_count/len_event   # Error percentage

                if error <= valid_event_on:
                    valid += 1

            detection = valid/event_count

        else:
            if sensitivity == 1:
                detection = 1

        detection_on,valid_on,event_count_on = [detection,valid,event_count]

        begin = (edges < 0).nonzero()[0]
        end = (edges > 0).nonzero()[0]

        if true[0] == 0:
            begin = np.insert(begin,0,0)

        if true[-1] == 0:
            end = np.append(end,n)

        event_count = len(begin)   # Number of events in true
        valid = 0
        detection = 0
        long_count = 0
        long_detect = 0
        long_detection = 0
        if event_count > 0:
            for i in range(event_count):
                len_event = end[i]-begin[i]
                if len_event > long_event:
                    long_count += 1
                    
                error_count = sum(predicted[begin[i]:end[i]])
                error = error_count/len_event   # Error percentage

                if error <= valid_event_off:
                    valid += 1
                    if len_event > long_event:
                        long_detect += 1

            detection = valid/event_count
            if long_count > 0:
                long_detection = long_detect/long_count
            else:
                long_detection = 1
                
        else:
            if specificity == 1:
                detection = 1
            long_detection = 1

        detection_off,valid_off,event_count_off = [detection,valid,event_count]

        not_found = event_count_off - valid_off

        if ofs_locate:
            return [accuracy, sensitivity, specificity, precision, negative_predictive_value, (detection_on,valid_on,event_count_on), (detection_off,valid_off,event_count_off), over_found, over_found_sleep,long_detection, ofs_location, of_location]
        else:
            return [accuracy, sensitivity, specificity, precision, negative_predictive_value, (detection_on,valid_on,event_count_on), (detection_off,valid_off,event_count_off), over_found, over_found_sleep,long_detection]


# Processes data from read actigraphic log and outputs DataFrame
def log_to_df(log,ws=11,return_all=False):
    # log is the input
    # ws is the size of the window used in variance and covariance computations
    # return_all is a boolean. If true, all calculated variables are returned.

    int_temp = log.temperature
    ext_temp = log.ext_temperature
    offwrist = np.where(log.state == 4, 0, 1)

    datetime = log.timestamps
    df = pd.DataFrame(datetime,columns=["datetime"])

    # All numerical data is scaled beforehand
    act = scale_by_max(log.pim)
    df["pim"] = act 
    df["zcm"] = scale_by_max(log.zcm)

    pad_size = int((ws-1)/2)   # Inputs will be padded before variance is calculated

    df["int_temp"] = scale_by_max(int_temp)
    # pad = df["int_temp"].mean()*np.ones(pad_size)
    # padded = pd.Series(np.concatenate((pad, df["int_temp"].to_numpy(), pad)))   # Padded signal
    # int_var = padded.rolling(ws,center=True).var().to_numpy()[pad_size:(len(padded)-pad_size)]   # Variance is calculated using padded signal

    df["ext_temp"] = (1/max(log.ext_temperature))*np.array(log.ext_temperature)
    # pad = df["ext_temp"].mean()*np.ones(pad_size)
    # padded_aux = pd.Series(np.concatenate((pad, df["ext_temp"].to_numpy(), pad)))
    # ext_var = padded_aux.rolling(ws,center=True).var().to_numpy()[pad_size:(len(padded_aux)-pad_size)]

    # temp_cov = padded.rolling(ws,center=True).cov(padded_aux).to_numpy()[pad_size:(len(padded)-pad_size)]   # Covariance is calculated between both padded signals

    # df["temp_cov"] = scale_by_max(temp_cov)

    # df["int_var"] = scale_by_max(int_var)

    # df["ext_var"] = scale_by_max(ext_var)

    df["dint_temp"] = scale_by_max(np.concatenate(([0],np.diff(int_temp))))
    
    df["dext_temp"] = scale_by_max(np.concatenate(([0],np.diff(ext_temp))))

    df["dif_temp"] = scale_by_max(np.subtract(int_temp,ext_temp))

    diff = np.diff(np.concatenate(([0],int_temp)))
    me_diff = mean_filter(diff,2,padding="same")

    df["diff_temp"] = scale_by_max(np.abs(me_diff))

    df["offwrist"] = offwrist

    if return_all:
        return act, int_temp, ext_temp, int_var, ext_var, temp_cov, offwrist, df

    else:
        return df


# Processes data so that it can be fed to the neural network both for training and testing
def model_data_process(df,ws=20,even=True,cols=['pim','int_temp','ext_temp'],ans="offwrist",use_str=False):
    # df is the input DataFrame
    # ws is the number of timesteps used in each sample
    # If even is True, every sample will be composed of a centering epoch, ws preceding epochs and ws succeeding epochs. 
    # If even is False, every sample will be composed of an epoch and the ws succeeding epochs.
    # cols defines wich columns of the DataFrame will be used in a sample
    # ans is the name of the column containing the desired outcomes. If ans==None, the function only outputs samples.

    n = len(df)    # Number of inputs in the DataFrame
    features = len(cols)   # Number of features
    index_offset = int(df.index[0])
    x = []
    y = []

    if ans is None:
        if even:
            for i in range(index_offset+ws,index_offset+n-ws):
                start = i-ws
                end = i+ws
                if use_str:
                    start = str(start)
                    end = str(end)
                    
                x.append(df.loc[start:end,cols].to_numpy().reshape(2*ws+1,features))

        else:
            for i in range(index_offset,index_offset+n-ws):
                start = i
                end = i+ws
                if use_str:
                    start = str(start)
                    end = str(end)

                x.append(df.loc[start:end,cols].to_numpy().reshape(ws+1,features))

        return np.array(x)

    else:
        if even:
            for i in range(index_offset+ws,index_offset+n-ws):
                start = i-ws
                end = i+ws
                y_index = i
                if use_str:
                    start = str(start)
                    end = str(end)
                    y_index = str(y_index)

                x.append(df.loc[start:end,cols].to_numpy().reshape(2*ws+1,features))
                y.append(df.at[y_index,ans])

        else:
            for i in range(index_offset,index_offset+n-ws):
                start = i
                end = i+ws
                y_index = i
                if use_str:
                    start = str(start)
                    end = str(end)
                    y_index = str(y_index)

                x.append(df.loc[start:end,cols].to_numpy().reshape(ws+1,features))
                y.append(df.at[y_index,ans])

        return np.array(x),np.array(y)