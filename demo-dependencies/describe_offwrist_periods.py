import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, auc
import jax.numpy as jnp
from datetime import date,datetime,timedelta,time
from scipy.signal import find_peaks as peak

import sys,inspect,os

from functions import segmentation,zero_prop,below_prop,var_filter


def describe_offwrist_periods(offwrist_periods,activity,temperature,temperature_variance,datetime_stamps,temperature_threshold,activity_threshold,refined_low_activity_threshold,additional_information=[],true_offwrist=[],is_lumus_file=False,segments=7,window_length=60,verbose=False,save=False,filename='offwrist_description.csv'):
    '''Computes several descriptive features of the input offwrist 
    periods.


    Parameters
    ----------
    offwrist_periods : list or np.array
                In the form [[start0,end0],...,[startN,endN]]
    activity : np.array [float]
            Read activity information
    temperature : np.array [float]
            Read temperature information
    temperature_variance : np.array [float]
            Temperature variance computed from read temperature
    datetime_stamps : np.array [datetime stamp]
            datetime.datetime stamps read by the device
    temperature_threshold : float
            Estimated temperature threshold that separates offwrist
            and onwrist periods. Temperature levels above this thre-
            shold indicate onwrist
    áctivity_threshold : float
            Estimated temperature threshold that separates offwrist
            and onwrist periods. Temperature levels above this thre-
            shold indicate onwrist
    activity_median_low : np.array [int]
            1 indicates low activity median, 0 otherwise
    window_length : int
            Length of the window used to compute the features rela-
            ting the activity and temperature levels around the off-
            wrist period
    additional_information : np.array [object]
            Additional information read by device-specific sensors,
            e.g. capacitive sensors in the ActLumus
    is_lumus_file : boolean
            If True, data was read from ActLumus device
    segments : int
            Number of segments to divide the offwrist period in when
            computing weight-related features
    window_length : int
            Length of the window used to compute the features rela-
            ting the activity and temperature levels around the off-
            wrist period
    verbose: boolean
            Verbosity level
    save: boolean
            If True, the resulting report will be saved to a file
    filename: string
            When save==True, the name of the file where the report
            will be written
    
    Returns
    -------
    offwrist_description : pd.DataFrame
            Description report for the offwrist periods
    '''
    data_length = len(activity)
    offwrist_description = pd.DataFrame(index=np.arange(len(offwrist_periods)),columns=["start","end","length","activity_zero_proportion","low_activity_proportion","low_temperature_proportion","start_activity_weight","end_activity_weight","border_activity_concentration","start_temperature_weight","end_temperature_weight","border_temperature_concentration"])

    additional_information_available = False
    if len(additional_information) > 0:
        additional_information_available = True

    # If datetime stamps aren't passed as parameters, a dummy array
    # is created to make the datetime computations possible
    if len(datetime_stamps) == 0:
        start = datetime.today()
        datetime_stamps = [start + i*timedelta(minutes=1) for i in range(data_length)]

    offwrist_description["temperature_difference_median"] = 0.0
    offwrist_description["temperature_difference_variance"] = 0.0
    offwrist_description["capacitive_difference_variance"] = 0.0
    if additional_information_available:
        if is_lumus_file:
            capacitive_sensor1 = additional_information["c1"]
            capacitive_sensor2 = additional_information["c2"]
            capacitive_sensor_difference = additional_information["capsensor_dif"]
        else:
            temperature_difference = additional_information["dif"]
            temperature_difference_variance = var_filter(temperature_difference,3)

    offwrist_description["high_activity_proportion_before"] = 0.0
    offwrist_description["high_activity_proportion_after"] = 0.0
    offwrist_description["low_activity_proportion"] = 0.0

    true_offwrist_available = False
    if len(true_offwrist) > 0:
        true_offwrist_available = True

        detected_offwrist = np.ones(data_length)
        for offwrist_index in offwrist_periods:
            detected_offwrist[offwrist_index[0]:offwrist_index[1]] = 0.0

        offwrist_description["true_offwrist"] = False

    activity_zero_proportion = zero_prop(activity)
    activity_threshold_quantile = activity_zero_proportion + (1.0 - activity_zero_proportion)*0.05
    activity_threshold = np.quantile(activity,activity_threshold_quantile,method="inverted_cdf")

    offwrist_index = 0
    for off in offwrist_periods:
        period_length = off[1]-off[0]
        period_activity = activity[off[0]:off[1]]
        period_temperature = temperature[off[0]:off[1]]
        period_temperature_variance = temperature_variance[off[0]:off[1]]

        if true_offwrist_available:
            true_offwrist_period = true_offwrist[off[0]:off[1]]
            error = np.sum(true_offwrist_period)/period_length
            if error < 0.1:
                offwrist_description.at[offwrist_index,"true_offwrist"] = True


        # The segmentation module computes the percentage of the to-
        # tal quantity in each segment of the input
        # Example: a=[1,2,3,4] ; sum(a)==10
        #          segmentation(a,2)==[0.3,0.7]
        activity_segmentation = segmentation(period_activity,segments)
        temperature_variance_segmentation = segmentation(period_temperature_variance,segments)

        offwrist_description.at[offwrist_index,"start"] = datetime_stamps[off[0]]
        if off[1] < data_length:
            offwrist_description.at[offwrist_index,"end"] = datetime_stamps[off[1]]
            # Period length is calculated in minutes using the given
            # datetime stamps
            offwrist_description.at[offwrist_index,"length"] = (datetime_stamps[off[1]] - datetime_stamps[off[0]]).total_seconds()/60.0
        else:
            offwrist_description.at[offwrist_index,"end"] = datetime_stamps[off[1]-1]
            offwrist_description.at[offwrist_index,"length"] = (datetime_stamps[off[1]-1] - datetime_stamps[off[0]]).total_seconds()/60.0
        
        offwrist_description.at[offwrist_index,"activity_zero_proportion"] = zero_prop(period_activity)
        offwrist_description.at[offwrist_index,"low_activity_proportion"] = below_prop(period_activity,refined_low_activity_threshold)

        # Ideal offwrists have high levels of activity before their
        # start and after their end.
        if off[0] >= window_length:
            high_activity_proportion_before = 1-below_prop(activity[off[0]-window_length:off[0]],activity_threshold)
        else:
            high_activity_proportion_before = 1-below_prop(activity[0:off[0]],activity_threshold)
        offwrist_description.at[offwrist_index,"high_activity_proportion_before"] = high_activity_proportion_before

        if off[1]+window_length <= data_length:
            high_activity_proportion_after = 1-below_prop(activity[off[1]:off[1]+window_length],activity_threshold)
        else:
            high_activity_proportion_after = 1-below_prop(activity[off[1]:data_length],activity_threshold)
        offwrist_description.at[offwrist_index,"high_activity_proportion_after"] = high_activity_proportion_after

        # Ideal offwrist periods have low levels of activity and tem-
        # perature inside, but if they do contain high levels of acti-
        # vity or temperature, it must be concentrated in the borders
        # relating to the transition process
        offwrist_description.at[offwrist_index,"start_activity_weight"] = activity_segmentation[0]
        offwrist_description.at[offwrist_index,"end_activity_weight"] = activity_segmentation[segments-1]
        offwrist_description.at[offwrist_index,"start_temperature_weight"] = temperature_variance_segmentation[0]
        offwrist_description.at[offwrist_index,"end_temperature_weight"] = temperature_variance_segmentation[segments-1]

        
        offwrist_description.at[offwrist_index,"low_temperature_proportion"] = below_prop(period_temperature,temperature_threshold)

        if additional_information_available:
            if is_lumus_file:
                period_capacitive_sensor1 = capacitive_sensor1[off[0]:off[1]]
                period_capacitive_sensor2 = capacitive_sensor2[off[0]:off[1]]
                period_capacitive_sensor_difference = capacitive_sensor_difference[off[0]:off[1]]
                offwrist_description.at[offwrist_index,"capacitive_difference_variance"] = np.var(period_capacitive_sensor_difference)
            else:
                period_temperature_difference = temperature_difference[off[0]:off[1]]
                period_temperature_difference_variance = temperature_difference_variance[off[0]:off[1]]
                offwrist_description.at[offwrist_index,"temperature_difference_median"] = np.median(period_temperature_difference)
                offwrist_description.at[offwrist_index,"temperature_difference_variance"] = np.median(period_temperature_difference_variance)

        offwrist_index += 1

    offwrist_description["border_activity_concentration"] = offwrist_description["start_activity_weight"] + offwrist_description["end_activity_weight"]
    offwrist_description["border_temperature_concentration"] = offwrist_description["start_temperature_weight"] + offwrist_description["end_temperature_weight"]


    if verbose:
        print(offwrist_description)

    if save:
        offwrist_description.to_csv(path_or_buf="verbose/"+file+"_report.csv",sep=';',header=True,index_label=None,float_format="e")

    return offwrist_description