"""Actiware algorithms

Author: Julius A. P. P. de Paula (26/02/2024)
"""

import numpy as np
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)


def calculate_sleep_statistics(data,rest_intervals,sleep_column,sleep_onset_end_algorithm="epochs",activity_counts_column="",min_epochs=10,epoch_duration=60,add_timestamp=True,timestamp_column="DATE/TIME"):
    """Computes sleep statistics for all rest intervals (nights)

    Parameters
    ----------
    data : DataFrame
        Contains information of both Actiware and ActStudio devices 
    rest_intervals : DataFrame
        Contains the indices of the bed times and get up times of the rest 
        intervals contained in 'data'
    sleep_column : string
        Column to use for sleep information
    sleep_onset_end_algorithm : {"epochs","immobile"}
        Selects which algorithm to use for computing sleep onset and end
    min_epochs : int
        Sleep onset and end algorithms identify the first and the last group of 
        epochs of this length for which all epochs but one are scored as immobile
    activity_counts_column : string, default ""
        Optional. Column to use for activity counts information, only used if the
        Immobile time algorithm is chosen
    epoch_duration : {15,30,60,120}
        Optional, only used if the Immobile time algorithm is chosen. Device epoch 
        duration in seconds
    add_timestamp : boolean, default True
        Optional. If True, datetime stamps for the bed time and getuptime will be
        added to the output.
    timestamp_column : string, default "DATE/TIME"
        Optional, only used if add_timestamp==True. Column from which to extract
        datetime stamps.

    Returns
    -------
    sleep_statistics : DataFrame
       Nightly sleep statistics (TBT, TST, WASO, Latency, Innertia, Efficiency and #Awakenings)
    """

    sleep_statistics = pd.DataFrame(rest_intervals,columns=["bed_time","getup_time"])

    # Total bed time statistic for all nights
    sleep_statistics["tbt"] = sleep_statistics["getup_time"]-sleep_statistics["bed_time"]

    # Nightly sleep statistics
    for night in sleep_statistics.index:
        # Getting DateTime stamp infomation for bed and getup time
        if add_timestamp:
            sleep_statistics.at[night,"bed_time_dt"] = data.at[sleep_statistics.at[night,"bed_time"],timestamp_column]
            sleep_statistics.at[night,"getup_time_dt"] = data.at[sleep_statistics.at[night,"getup_time"],timestamp_column]

        # Sleep information (0 for asleep, 1 for awake) for the specific night
        night_sleep = data.loc[sleep_statistics.at[night,"bed_time"]:sleep_statistics.at[night,"getup_time"], sleep_column]

        # Latency and innertia information
        if sleep_onset_end_algorithm == "immobile":
            night_activity_counts = data.loc[sleep_statistics.at[night,"bed_time"]:sleep_statistics.at[night,"getup_time"], activity_counts_column]
            night_mobility = calculate_mobility(night_activity_counts,epoch_duration)
            onset,end,sleep_statistics.at[night,"latency"],sleep_statistics.at[night,"innertia"] = immobile_sleep_onset_end(night_mobility,min_epochs)
        else:
            onset,end,sleep_statistics.at[night,"latency"],sleep_statistics.at[night,"innertia"] = epochs_sleep_onset_end(night_sleep,min_epochs)

        # Wake after sleep onset statistic
        sleep_statistics.at[night,"waso"] = np.sum(night_sleep[onset:end])

        # Number of awakenings statistic
        sleep_statistics.at[night,"awakenings"] = get_awakenings(night_sleep[onset:end])
    
    # Total sleep time statistic for all nights
    sleep_statistics["tst"] = sleep_statistics["tbt"]-sleep_statistics["waso"]-sleep_statistics["latency"]-sleep_statistics["innertia"]

    # Sleep efficiency statistic for all nightss
    sleep_statistics["efficiency"] = sleep_statistics["tst"]/sleep_statistics["tbt"]

    return sleep_statistics

def get_rest_intervals(state):
    """Gets the indices of the start and the end of the rest intervals in
       a STATE column from an ActStudio device

    Parameters
    ----------
    state : np.array
        STATE information (0 for awake, 1 for sleep, 2 for rest)

    Returns
    -------
    rest_intervals : DataFrame
        Contains the indices of the bed times and get up times of the rest 
        intervals contained in 'data'
    """

    edges = np.diff(np.where(state > 0,1,0))
    edges_index = edges.nonzero()[0].astype(int)  # Output edges

    rest_intervals = edges_index.reshape((int(len(edges_index)/2),2))

    return rest_intervals

def immobile_sleep_onset_end(night_mobility,min_epochs):
    """Implementation of the Actiware algorithm for computing sleep onset and 
       sleep end from mobility

    Parameters
    ----------
    night_mobility : np.array
        Mobility information (1 for mobile, 0 for immobile)
    min_epochs : int
        The algorithm identifies the first and the group of epochs of this 
        length for which all epochs but one are scored as immobile

    Returns
    -------
    onset : int
        Sleep onset index
    end : int
        Sleep end index
    latency : int
        Number of epochs between bed time and sleep onset
    innertia : int
        Number of epochs between sleep end and getup time
    """

    # Computing the sum of the mobility scores for groups of epochs with length min_epochs
    onset_rolling_sum = np.convolve(night_mobility,np.ones(min_epochs),mode="valid")
    # We're interested in groups where the sum is 1, i.e. all epochs but one are scored as 
    # immobile
    onset_candidates = np.where(onset_rolling_sum == 0,1,0)
    # Using np.nonzero() we select the indices of the groups with sum equal to 1
    onset_candidates_indices = onset_candidates.nonzero()[0]
    # And the first one is selected as the onset
    if len(onset_candidates_indices > 0):
        onset = int(onset_candidates_indices[0])
    else:
        onset = 0
    latency = onset

    # To compute sleep end we'll apply the same algorithm but the mobility array is mirrored
    # horizontally, so we'll start searching from the last epoch
    end_rolling_sum = np.convolve(np.flip(night_mobility),np.ones(min_epochs),mode="valid")
    end_candidates = np.where(end_rolling_sum == 0,1,0)
    end_candidates_indices = end_candidates.nonzero()[0]
    # In this case, the index of first group is the distance between the get up time (last 
    # epoch) to the first epoch of actual sleep, that is the definition of sleep innertia
    if len(end_candidates_indices > 0):
        innertia = int(end_candidates_indices[0])
    else:
        innertia = 0
    # Sleep end is the index of the last epoch of actual sleep
    end = len(night_mobility)-innertia

    return onset,end,latency,innertia

def epochs_sleep_onset_end(night_sleep,min_epochs):
    """Implementation of the Actiware algorithm for computing sleep onset and 
       sleep end from epoch's score

    Parameters
    ----------
    night_sleep : np.array
        SLEEP information (0 for asleep, 1 for awake)
    min_epochs : int
        The algorithm identifies the first and the last group of epochs scored as sleep
        that is at least this number of epochs in length

    Returns
    -------
    onset : int
        Sleep onset index
    end : int
        Sleep end index
    latency : int
        Number of epochs between bed time and sleep onset
    innertia : int
        Number of epochs between sleep end and getup time
    """
    
    # Computing the sum of the sleep scores for groups of epochs with length min_epochs
    onset_rolling_sum = np.convolve(night_sleep,np.ones(min_epochs),mode="valid")
    # We're interested in groups where the sum is 0, i.e. all epochs are sleep
    onset_candidates = np.where(onset_rolling_sum == 0,1,0)
    # Using np.nonzero() we select the indices of the groups with zero sum
    onset_candidates_indices = onset_candidates.nonzero()[0]
    # And the first one is selected as the onset
    if len(onset_candidates_indices > 0):
        onset = int(onset_candidates_indices[0])
    else:
        onset = 0    
    latency = onset

    # To compute sleep end we'll apply the same algorithm but the sleep array is mirrored
    # horizontally, so we'll start searching from the last epoch
    end_rolling_sum = np.convolve(np.flip(night_sleep),np.ones(min_epochs),mode="valid")
    end_candidates = np.where(end_rolling_sum == 0,1,0)
    end_candidates_indices = end_candidates.nonzero()[0]
    # In this case, the index of first group is the distance between the get up time (last 
    # epoch) to the first epoch of actual sleep, that is the definition of sleep innertia
    if len(end_candidates_indices > 0):
        innertia = int(end_candidates_indices[0])
    else:
        innertia = 0    
    # Sleep end is the index of the last epoch of actual sleep
    end = len(night_sleep)-innertia

    return onset,end,latency,innertia


def get_awakenings(night_sleep):
    """Calculates the number of awakenings statistic from SLEEP information
       of a single night

    Parameters
    ----------
    night_sleep : np.array
        SLEEP information (0 for asleep, 1 for awake)

    Returns
    -------
    awakenings : int
       Count of the awakenings of a patient in a night's sleep
    """

    edges = np.diff(night_sleep)
    awakenings = int(np.sum(np.where(edges > 0,1,0)))
    return awakenings


def calculate_activity_counts(pim,activity_factor=0.0656):
    return activity_factor*pim


def calculate_sleep(activity_counts,wake_threshold_selection,epoch_duration,custom_value=0.0):
    """Computes Sleep/Wake information using ActiWare weighted sum algorithm

    Parameters
    ----------
    activity_counts : np.array
        Activity counts read by the device
    wake_threshold_selection : {"low","medium","high","auto","custom"}
        The wake threshold value will be selected based on this parameter
    epoch_duration : {15,30,60,120}
        Device epoch duration in seconds
    custom_value : float, default 0.0
        Custom value for the wake threshold value, only used if
        wake_threshold_selection=="custom"

    Returns
    -------
    sleep : np.array
       Sleep/Wake information from total activity counts (0 means subject asleep)
    """

    if wake_threshold_selection == "auto":
        custom_value = calculate_auto_threshold_value(activity_counts,epoch_duration)
    wake_threshold_value = select_wake_threshold_value(wake_threshold_selection,custom_value)

    total_activity_counts = calculate_total_activity_counts(activity_counts,epoch_duration)

    return np.where(total_activity_counts >= wake_threshold_value, 1, 0)

def select_wake_threshold_value(wake_threshold_selection,custom_value):
    """Selects wake threshold value based on user choice 

    Parameters
    ----------
    wake_threshold_selection : {"low","medium","high","auto","custom"}
        The wake threshold value will be selected based on this parameter
    custom_value : float
        Custom value for the wake threshold value. Also used here to store
        calculated automatic wake threshold value

    Returns
    -------
    wake_threshold_value : float
        Epochs with total activity counts greater than or equal to this
        value are scored as wake
    """

    if wake_threshold_selection == "low":
        return 20.0

    elif wake_threshold_selection == "medium":
        return 40.0

    elif wake_threshold_selection == "high":
        return 80.0

    else:
        return custom_value

def calculate_auto_threshold_value(activity_counts,epoch_duration):
    """Calculates automatic wake threshold value based on percentage of
        mobile time

    Parameters
    ----------
    activity_counts : np.array
        Activity counts read by the device
    epoch_duration : {15,30,60,120}
        Device epoch duration in seconds

    Returns
    -------
    auto_threshold_value : float
        Automatic wake threshold value 
    """

    mobility = calculate_mobility(activity_counts,epoch_duration)
    mobile_time = np.sum(mobility)

    return 0.88888*(np.sum(activity_counts)/mobile_time)

def calculate_mobility(activity_counts,epoch_duration):
    """Calculates mobility information from activity counts

    Parameters
    ----------
    activity_counts : np.array
        Activity counts read by the device
    epoch_duration : {15,30,60,120}
        Device epoch duration in seconds

    Returns
    -------
    mobility : np.array
        Mobility information (1 means subject mobile)
    """

    mobile_count = epoch_duration/15

    return np.where(activity_counts >= mobile_count, 1, 0)

def calculate_total_activity_counts(activity_counts,epoch_duration):
    """Calculates total activity counts using the weighted sum
       algorithm with weights selected based on epoch duration

    Parameters
    ----------
    activity_counts : np.array
        Activity counts read by the device
    epoch_duration : {15,30,60,120}
        Device epoch duration in seconds

    Returns
    -------
    total_activity_counts : np.array
        Total activity counts from activity counts
    """

    weights,half_window_size = select_weights(epoch_duration)

    # The signal is paddded so that the weights window will fit the first
    # and last epoch
    padded_activity_counts = pad_signal(activity_counts,half_window_size)
    
    # np.convolve is used for the rolling weighted sum
    return np.convolve(padded_activity_counts,weights,mode="valid")

def select_weights(epoch_duration):
    """Select weights based on epoch duration

    Parameters
    ----------
    epoch_duration : {15,30,60,120}
        Device epoch duration in seconds

    Returns
    -------
    weights : np.array
        Window of weights
    half_window_size : int
        Number of previous or following epochs used in the window
    """

    if epoch_duration == 15:
        return np.array([0.04,0.04,0.04,0.04, 0.2,0.2,0.2,0.2, 4, 0.2,0.2,0.2,0.2, 0.04,0.04,0.04,0.04,]), 8

    elif epoch_duration == 30:
        return np.array([0.04,0.04, 0.2,0.2, 2, 0.2,0.2, 0.04,0.04,]), 4

    elif epoch_duration == 60:
        return np.array([0.04, 0.2, 1, 0.2, 0.04,]), 2

    else:
        return np.array([0.12, 0.5, 0.12,]), 1

def pad_signal(signal,pad_size):
    """Pads a signal in the beginning and the end with the respective value

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

    beginning_pad = signal[0]*np.ones(pad_size)
    ending_pad = signal[signal_length-1]*np.ones(pad_size)

    padded_signal = np.insert(signal.copy(),0,beginning_pad)
    padded_signal = np.append(padded_signal,ending_pad)

    return padded_signal

# activity_counts = np.array([1,1,1,2,3,4])
# print(calculate_total_activity_counts(activity_counts,60))