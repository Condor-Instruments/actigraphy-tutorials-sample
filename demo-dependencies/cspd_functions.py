# -*- coding: utf-8 -*-

# CSPD algorithm: helping functions - 2021
# Julius Andretti

import os,sys,inspect
os.environ['NUMPY_EXPERIMENTAL_ARRAY_FUNCTION'] = '0'
import cProfile, pstats, io
import numpy as np
import pandas as pd
from datetime import date,datetime,timedelta,time
from scipy.signal import find_peaks as peak  


from functions import *

def compute_features(activity,
                     threshold,
                     start,
                     end,
                     ):
        """Compute features of the activity inside a peak or valley.

        Parameters
        ----------
        activity : np.array [float]
                Read activity information.
        threshold : float
            Median activity levels above this parameter will be classi-
            fied as high.
        start : int
                Index of first epoch of a peak or valley
        end : int
                Index of last epoch of a peak or valley

        Returns
        -------
        length : int
                Length of the peak/valley in epochs.
        mean_activity : float
                Mean activity inside the peak/valley.
        median_activity : float
                Median activity inside the peak/valley.
        zero_proportion : float
                Proportion of zeros inside the peak/valley.
        above_threshold_proportion : float
                Proportion of epochs with a 'high' level of activity 
                inside the peak/valley.
        """

        region_activity = activity[start:end]
        length = end-start
        mean_activity = np.mean(region_activity)
        median_activity = np.median(region_activity)
        zero_proportion = zero_prop(region_activity)
        above_threshold_proportion = sum(np.where(region_activity > threshold,1,0))/(end-start)

        return length,mean_activity,median_activity,zero_proportion,above_threshold_proportion

def identify_peaks_and_valleys(activity,median_activity,threshold):
    """Applies a thresholding operation to the input median activity to
    identify regions with a sustained low ("valley") or high ("peak") 
    activity level.

    Parameters
    ----------
    activity : np.array [float]
            Read activity information.
    median_activity : np.array [float]
            Median activity computed from read activity using a short
            filter window.
    threshold : float
            Median activity levels above this parameter will be classi-
            fied as high.

    Returns
    -------
    peaks_and_valleys: pd.DataFrame
            Contains the information of the location and features of 
            peaks and valleys inside a given signal.
    """

    data_length = len(activity)
    columns = ["class","start","end","length","mean","median","zero_proportion","above_threshold_proportion"]
    peaks_and_valleys = pd.DataFrame()
    
    thresholded_activity = np.where(median_activity > threshold, 1, 0)

    t = 0
    while t < data_length:
        # First epoch of a sequence of high median activity.
        if thresholded_activity[t]:
            start = t

            # This loop goes to the end of the sequence.
            t += 1
            while (t < data_length) and (thresholded_activity[t]):
                t += 1
            end = t

            region_class = "p"

        else:
            start = t

            t += 1
            while (t < data_length) and (not thresholded_activity[t]):
                t += 1
            end = t

            region_class = "v"

        # Features of the peak/valley are then computed and stored.
        length,mean,median,zero_proportion,above_threshold_proportion = compute_features(activity,threshold,start,end)
        dictionary = {"class":region_class,
                      "start":start,
                      "end":end,
                      "length":length,
                      "mean":mean,
                      "median":median,
                      "zero_proportion":zero_proportion,
                      "above_threshold_proportion":above_threshold_proportion,}

        peaks_and_valleys = pd.concat([peaks_and_valleys,pd.DataFrame(dictionary,index=[0])],axis=0,ignore_index=True)

    return peaks_and_valleys.reindex(columns=columns)

def remove_peak_valley(peaks_and_valleys,index_to_remove,activity,threshold):
    """Merges a peak or valley into the neighboring regions to create a 
    large new region composed by 2 or 3 regions. 

    Parameters
    ----------
    peaks_and_valleys: pd.DataFrame
            Contains the information of the location and features of 
            peaks and valleys inside a given signal.
    index_to_remove : int
            Index of the peak or valley to be removed.
    activity : np.array [float]
            Read activity information.
    threshold : float
            Median activity levels above this parameter will be classi-
            fied as high.

    Returns
    -------
    peaks_and_valleys: pd.DataFrame
            Contains the information of the location and features of 
            peaks and valleys inside a given signal.
    """

    peaks_and_valleys_count = len(peaks_and_valleys)

    if index_to_remove == 0:
        # If the first region is to be removed, it is merged with the
        # second one.
        start = int(peaks_and_valleys.at[0,"start"])
        end = int(peaks_and_valleys.at[1,"end"])

        peaks_and_valleys.at[1,"start"] = start
        peaks_and_valleys.loc[1,["length","mean","median","zero_proportion","above_threshold_proportion"]] = compute_features(activity,threshold,start,end)

        peaks_and_valleys.drop(index=[0],inplace=True)
        
    else:
        start = int(peaks_and_valleys.at[index_to_remove-1,"start"])

        if index_to_remove < peaks_and_valleys_count-1:
            end = int(peaks_and_valleys.at[index_to_remove+1,"end"])
            peaks_and_valleys.drop(index=[index_to_remove,index_to_remove+1],inplace=True)

        else:
            # If the last region is to be removed, it is merged with
            # the penultimate one.
            end = int(peaks_and_valleys.at[index_to_remove,"end"])
            peaks_and_valleys.drop(index=[index_to_remove],inplace=True)

        peaks_and_valleys.at[index_to_remove-1,"end"] = end
        peaks_and_valleys.loc[index_to_remove-1,["length","mean","median","zero_proportion","above_threshold_proportion"]] = compute_features(activity,threshold,start,end,)


    # After the removal, the index is updated.
    peaks_and_valleys_count = len(peaks_and_valleys)
    peaks_and_valleys.index = range(peaks_and_valleys_count)

    return peaks_and_valleys

def boolean_length_filter(minimum_length,signal,class_to_filter="v"):
    """Removes from an array of 0s and 1s the sequences that are too short.

    Parameters
    ----------
    minimum_length : int
            Sequences shorter than this parameter will be filtered out.
    signal : np.array [float]
            A thresholding operation will be applied to this data to 
            compute the peaks and valleys.
    class_to_filter : {"p","v"}
            If "p", sequences of 1s will be filtered out. If "v", sequen-
            ces of 0s will be filtered out.
    threshold : float
            Signal levels above this parameter will be classified as high.

    Returns
    -------
    filtered_signal : np.array [float]
            Input array with short sequences filtered out.
    """

    data_length = len(signal)

    # A zero activity will be used here because, since this module handles
    # a boolean input, it doesn't use the computed features.
    activity = np.zeros(data_length)
    threshold = 0.5

    peaks_and_valleys = identify_peaks_and_valleys(activity,signal,threshold)

    peaks_and_valleys_count = len(peaks_and_valleys)

    filtered_signal = np.ones(data_length)
    if peaks_and_valleys_count > 1:
        t = 0
        while (t < peaks_and_valleys_count):
            remove = False
            if (peaks_and_valleys.at[t,"length"] <= minimum_length) and (peaks_and_valleys.at[t,"class"] == class_to_filter):
                remove = True

            if remove:
                peaks_and_valleys = remove_peak_valley(peaks_and_valleys,t,activity,threshold)
                peaks_and_valleys_count = len(peaks_and_valleys)

            else:
                t += 1


        # After all short sequences are filtered, the filtered signal is
        # constructed.
        peaks_and_valleys_count = len(peaks_and_valleys)
        t = 0
        while (t < peaks_and_valleys_count):
            if (peaks_and_valleys.at[t,"class"] == "v"):
                filtered_signal[int(peaks_and_valleys.at[t,"start"]):int(peaks_and_valleys.at[t,"end"])] = 0
            t += 1

        # The signal is filtered for a second time to prevent errors.
        peaks_and_valleys = identify_peaks_and_valleys(activity,filtered_signal,threshold)
        peaks_and_valleys_count = len(peaks_and_valleys)

        if peaks_and_valleys_count > 1:
            it = 0
            while (t < peaks_and_valleys_count):
                remove = False
                if (peaks_and_valleys.at[t,"length"] <= minimum_length) and (peaks_and_valleys.at[t,"class"] == class_to_filter):
                    remove = True

                if remove:
                    # Transition filtering process takes place
                    peaks_and_valleys = remove_peak_valley(peaks_and_valleys,t,activity,threshold)
                    peaks_and_valleys_count = len(peaks_and_valleys)

                else:
                    t += 1

            filtered_signal = np.ones(data_length)
            peaks_and_valleys_count = len(peaks_and_valleys)
            for t in range(peaks_and_valleys_count):
                if (peaks_and_valleys.at[t,"class"] == "v"):
                    filtered_signal[int(peaks_and_valleys.at[t,"start"]):int(peaks_and_valleys.at[t,"end"])] = 0

    elif peaks_and_valleys_count == 1:
        filtered_signal = signal

    return filtered_signal

def peak_valley_zero_proportion_filter(minimum_zero_proportion,signal,class_to_filter="v"):
    """Removes from an array of 0s and 1s the sequences that are don't have a 
    high enough proportion of zeros.

    Parameters
    ----------
    minimum_length : int
            Sequences with a proportion of zeros smaller than this parame-
            ter will be filtered out.
    signal : np.array [float]
            A thresholding operation will be applied to this data to 
            compute the peaks and valleys.
    class_to_filter : {"p","v"}
            If "p", sequences of 1s will be filtered out. If "v", sequen-
            ces of 0s will be filtered out.
    threshold : float
            Signal levels above this parameter will be classified as high.

    Returns
    -------
    filtered_signal : np.array [float]
            Input array with sequences that don't have a high enough propor-
            tion of zeros filtered out.
    """

    data_length = len(signal)

    activity = np.zeros(data_length)
    threshold = 0.5

    peaks_and_valleys = identify_peaks_and_valleys(activity,signal,threshold)

    peaks_and_valleys_count = len(peaks_and_valleys)

    filtered_signal = np.ones(data_length)

    if peaks_and_valleys_count > 1:
        t = 0
        while (t < peaks_and_valleys_count):
            remove = False

            if (peaks_and_valleys.at[t,"zero_proportion"] <= minimum_zero_proportion) and (peaks_and_valleys.at[t,"class"] == class_to_filter):
                remove = True

            if remove:
                peaks_and_valleys = remove_peak_valley(peaks_and_valleys,t,activity,threshold)
                peaks_and_valleys_count = len(peaks_and_valleys)

            else:
                t += 1

        # After all short sequences are filtered, the filtered signal is
        # constructed.
        peaks_and_valleys_count = len(peaks_and_valleys)
        t = 0
        while (t < peaks_and_valleys_count):
            if (peaks_and_valleys.at[t,"class"] == "v"):
                filtered_signal[int(peaks_and_valleys.at[t,"start"]):int(peaks_and_valleys.at[t,"end"])] = 0
            t += 1

        # The signal is filtered for a second time to prevent errors.
        peaks_and_valleys = identify_peaks_and_valleys(activity,filtered_signal,threshold)
        peaks_and_valleys_count = len(peaks_and_valleys)

        if peaks_and_valleys_count > 1:
            t = 0
            while (t < peaks_and_valleys_count):
                remove = False

                if (peaks_and_valleys.at[t,"zero_proportion"] <= minimum_zero_proportion) and (peaks_and_valleys.at[t,"class"] == class_to_filter):
                    remove = True

                if remove:
                    peaks_and_valleys = remove_peak_valley(peaks_and_valleys,t,activity,threshold)
                    peaks_and_valleys_count = len(peaks_and_valleys)

                else:
                    t += 1

            filtered_signal = np.ones(data_length)
            peaks_and_valleys_count = len(peaks_and_valleys)
            t = 0
            while (t < peaks_and_valleys_count):
                if (peaks_and_valleys.at[t,"class"] == "v"):
                    filtered_signal[int(peaks_and_valleys.at[t,"start"]):int(peaks_and_valleys.at[t,"end"])] = 0
                t += 1

    elif peaks_and_valleys_count == 1:
        filtered_signal = signal

    return filtered_signal