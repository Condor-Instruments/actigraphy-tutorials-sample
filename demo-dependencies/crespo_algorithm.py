# References:
# [1] CRESPO, C.; ABOY, M.; FERNÁNDEZ, J. R.; MOJÓN, A.: Automatic identification of activity-rest periods based on actigraphy (2012)

import numpy as np
import time, os, inspect, sys
import pandas as pd
from datetime import datetime, timedelta, date
from scipy.ndimage import binary_closing, binary_opening


from cspd_functions import *
from functions import *

np.set_printoptions(threshold=sys.maxsize)


class CrespoAlgorithm:
    def __init__(self,
                 consecutive_zeros_threshold=15,
                 awake_consecutive_zeros_threshold=2,
                 sleep_consecutive_zeros_threshold=30,
                 invalid_zeros_mitigation_percentile=0.33,
                 median_filter_window_hourly_length=8.0,
                 adaptive_median_filter_padding_hourly_length=1.0,
                 sleep_median_activity_quantile_threshold=1/3,
                 preprocessing_morphological_filter_structuring_element_size=61,
                 detect_naps=False,
                 nap_median_activity_threshold=2.0,
                 compute_output_naps_with_logical_and=False,
                 nap_zero_proportion_threshold=0.5,
                 nap_zero_proportion_filter_window_size=5,
                 apply_last_morphological_filter=False,
                 minimum_short_window_activity_median_threshold=1.0,
                 condition=0,
                 ):
        """Adds datetime.datetime information to a DataFrame containing the
        location of off or onwrist periods.

        Parameters
        ----------
        consecutive_zeros_threshold : int
                Threshold for consecutive zeroes.
        awake_consecutive_zeros_threshold : int
                Threshold for consecutive zeroes inside wake periods.
        sleep_consecutive_zeros_threshold : int
                Threshold for consecutive zeroes inside sleep periods.
        invalid_zeros_mitigation_percentile : float
                Percentile chosen to minimize the effect of invalid_zero_in-
                dexes zeroes during preprocessing.
        median_filter_window_hourly_length : float 
                Length in hours of the preprocessing median filter.
        adaptive_median_filter_padding_hourly_length : float
                Length in hours of the padding added prior to adaptive median 
                filter.
        sleep_median_activity_quantile_threshold : float
                Average number of hours of sleep.
        preprocessing_morphological_filter_structuring_element_size : int
                Window length for the preprocessing morphological filter.
        detect_naps : boolean
                If True, the algorithm searches for secondary sleep periods,
                also know as "naps".
                If False, the algorithm searches for primary sleep periods.
        nap_median_activity_threshold : float
                Epochs with a median activity level lower than this thre-
                shold may be scored as sleep.
        nap_zero_proportion_filter_window_size : int
                Size of the window used in the proportion of zeros filter.
        nap_zero_proportion_threshold : float
                Epochs with a zero-activity proportion around them that's low-
                er than this threshold may be scored as sleep.
        compute_output_naps_with_logical_and : boolean
                If True, epochs are scored as sleep if they have a low me-
                dian activity AND a high proportion of zeros.
                If False, epochs are scored as sleep if they have a low me-
                dian activity OR a high proportion of zeros.
        apply_last_morphological_filter : boolean
                If True,the last morphological operation is applied.
        minimum_short_window_activity_median_threshold : float
                Epochs with a short-window-median-filtered activity lower
                than this level may always be scored as sleep.
        condition : int
                User-set configuration. Used to indicate if the subject suf-
                fers from a particular condition.
        """

        self.consecutive_zeros_threshold = consecutive_zeros_threshold
        self.awake_consecutive_zeros_threshold = awake_consecutive_zeros_threshold
        self.sleep_consecutive_zeros_threshold = sleep_consecutive_zeros_threshold
        self.invalid_zeros_mitigation_percentile = invalid_zeros_mitigation_percentile
        self.median_filter_window_hourly_length = median_filter_window_hourly_length
        self.adaptive_median_filter_padding_hourly_length = adaptive_median_filter_padding_hourly_length
        self.sleep_median_activity_quantile_threshold = sleep_median_activity_quantile_threshold
        self.preprocessing_morphological_filter_structuring_element_size = preprocessing_morphological_filter_structuring_element_size
        self.detect_naps=detect_naps
        self.nap_median_activity_threshold=nap_median_activity_threshold
        self.compute_output_naps_with_logical_and=compute_output_naps_with_logical_and
        self.nap_zero_proportion_threshold=nap_zero_proportion_threshold
        self.nap_zero_proportion_filter_window_size=nap_zero_proportion_filter_window_size
        self.apply_last_morphological_filter=apply_last_morphological_filter
        self.minimum_short_window_activity_median_threshold=minimum_short_window_activity_median_threshold
        self.condition=condition


    def detect_msp(self,
                   activity,
                   datetime_stamps,
                   epoch_hour,
                   ):
        """Adds datetime.datetime information to a DataFrame containing the
        location of off or onwrist periods.

        Parameters
        ----------
        activity : np.array [float]
                input activity array
        datetime_stamps : np.array [datetime.datetime]
                datetime array containing the actigraphy timestamps
        epoch_hour : int
                Number of epochs correspondent to an 1 hour interval. 
        """

        self.datetime_stamps = datetime_stamps
        self.activity = activity

        # Number of points.
        data_length = len(self.activity)   

        # Filter window length.
        median_filter_window_size = int(epoch_hour*self.median_filter_window_hourly_length)+1
        # Filter half-window length.
        median_filter_half_window_size = int((median_filter_window_size-1)/2) 

        maximum_activity = np.max(self.activity)

        # Input array is padded at the beggining and at the end with a sequence of 
        # maximum_activity-valued elements with length of 
        # adaptive_median_filter_padding_hourly_length hours.
        pad_size = int(epoch_hour*self.adaptive_median_filter_padding_hourly_length)

        padded_activity = self.activity.copy()
        pad = maximum_activity*np.ones(pad_size)
        padded_activity = np.insert(padded_activity,0,pad)
        padded_activity = np.append(padded_activity,pad)

        if not self.detect_naps:   # Main sleep period detection, signal conditioning,
                                   # (i.e. the effect of long sequences of zero is miti-
                                   # gated by adding to them a low nonzero level of acti-
                                   # vity), is applied.
            mitigated_zeros_activity = self.activity.copy() 
            
            zero_mitigation_activity_level = np.quantile(self.activity,self.invalid_zeros_mitigation_percentile,interpolation='linear')   # Value of percentile t
            invalid_zeros_mask = np.ones(data_length)   # invalid_zeros_mask[i]==0 if i 
                                                        # is the index of an 
                                                        # invalid_zero_indexes zero.
            zero_sequence_length = 0
            for i in range(data_length):
                if self.activity[i] == 0.0:
                    zero_sequence_length += 1

                else:
                    if zero_sequence_length > self.consecutive_zeros_threshold:   # Sequences of more than consecutive_zeros_threshold consecutive zeroes are invalid_zero_indexes
                        mitigated_zeros_activity[i-zero_sequence_length:i] += zero_mitigation_activity_level
                        invalid_zeros_mask[i-zero_sequence_length:i] = 0

                    zero_sequence_length = 0

            # mitigated_zeros_activity is then padded at the beggining and at the end  
            # with a sequence of 60*median_filter_window_hourly_length elements of value 
            # maximum_activity=max(activity).

            print("data_length",data_length)
            print("mitigated_zeros_activity",len(mitigated_zeros_activity))

            pad = maximum_activity*np.ones(int(epoch_hour*self.median_filter_window_hourly_length))
            print("pad",len(pad))

            padded_mitigated_zeros_activity = np.insert(mitigated_zeros_activity,0,pad)
            padded_mitigated_zeros_activity = np.append(padded_mitigated_zeros_activity,pad)
            print("padded_mitigated_zeros_activity",len(padded_mitigated_zeros_activity))
            padded_mitigated_zeros_activity = median_filter(padded_mitigated_zeros_activity,median_filter_half_window_size,padding='padded')
            print("padded_mitigated_zeros_activity",len(padded_mitigated_zeros_activity))

            sleep_median_activity_threshold = np.quantile(padded_mitigated_zeros_activity,self.sleep_median_activity_quantile_threshold,interpolation='linear')
            
            # Thresholding operation.
            initial_sleep_detection = np.where(padded_mitigated_zeros_activity > sleep_median_activity_threshold, 1, 0)    
            print("initial_sleep_detection",len(initial_sleep_detection))

            # Morphological strutcturing element.
            structuring_element = np.ones(self.preprocessing_morphological_filter_structuring_element_size)   

            # Morphological operations.
            morphological_filtered_initial_detection = binary_opening(binary_closing(initial_sleep_detection,structuring_element),structuring_element).astype(int)
            print("morphological_filtered_initial_detection",len(morphological_filtered_initial_detection))

            invalid_zero_indexes = np.asanyarray([],dtype=int) # This array will contain 
                                                               # the indexes of the 
                                                               # invalid_zero_indexes ze-
                                                               # roes.
            awake_zero_sequence_length = 0
            sleep_zero_sequence_length = 0
            invalid_zeros_mask = np.ones(data_length)   # invalid_zeros_mask[i]==0 if i is 
                                                        # the index of an invalid zero.
            for i in range(data_length):
                if morphological_filtered_initial_detection[i]:   # If initially classified 
                                                                  # as awake.
                    if sleep_zero_sequence_length > self.sleep_consecutive_zeros_threshold:
                        invalid_zero_indexes = np.append(invalid_zero_indexes,range(i-sleep_zero_sequence_length,i))
                        invalid_zeros_mask[i-sleep_zero_sequence_length:i] = 0
                    sleep_zero_sequence_length = 0

                    if self.activity[i] == 0:
                        awake_zero_sequence_length += 1
                    
                    else:
                        if awake_zero_sequence_length > self.awake_consecutive_zeros_threshold:
                            invalid_zero_indexes = np.append(invalid_zero_indexes,range(i-awake_zero_sequence_length,i))
                            invalid_zeros_mask[i-awake_zero_sequence_length:i] = 0
                        awake_zero_sequence_length = 0

                else:   # If classified as sleep
                    if awake_zero_sequence_length > self.awake_consecutive_zeros_threshold:
                        invalid_zero_indexes = np.append(invalid_zero_indexes,range(i-awake_zero_sequence_length,i))
                        invalid_zeros_mask[i-awake_zero_sequence_length:i] = 0
                    awake_zero_sequence_length = 0
                    
                    if self.activity[i] == 0:
                        sleep_zero_sequence_length += 1
                    
                    else:
                        if sleep_zero_sequence_length > self.sleep_consecutive_zeros_threshold:
                            invalid_zero_indexes = np.append(invalid_zero_indexes,range(i-sleep_zero_sequence_length,i))
                            invalid_zeros_mask[i-sleep_zero_sequence_length:i] = 0
                        sleep_zero_sequence_length = 0

            invalid_zero_indexes = np.array(invalid_zero_indexes)+pad_size   # Move invalid 
                                                                             # indexes to 
                                                                             # match the ones
                                                                             # in the padded 
                                                                             # array.
            padded_activity[invalid_zero_indexes] = np.nan   # Invalid points are marked as NaN.

            activity_zero_proportions = None
            thresholded_activity_zero_proportions = None
            thresholded_median_activity = None

        else:
            padded_mitigated_zeros_activity = None
            initial_sleep_detection = None
            morphological_filtered_initial_detection = None

        # Median filtering
        adaptive_median_filtered_activity = np.zeros(data_length)

        minimum_half_window_size = pad_size
        maximum_half_window_size = median_filter_half_window_size
        variable_window_size = True   # If True, window size progressively increases and then 
                                      # progressively decreases.
                                      # If False, window size is constant and equal to 
                                      # maximum_half_window_size.

        half_window_size = maximum_half_window_size
        if variable_window_size:
            half_window_size = minimum_half_window_size
            
        for i in range(data_length):
            # Median window center.
            center = i+pad_size   

            # Take the median in desired window ignoring NaN values.
            adaptive_median_filtered_activity[i] = np.nanmedian(padded_activity[center-half_window_size:center+half_window_size+1])   
            
            # If the median activity comes out NaN, the closest pre-
            # vious numeric value is taken.
            if np.isnan(adaptive_median_filtered_activity[i]):
                if i > 0:
                    j = i-1
                    while np.isnan(adaptive_median_filtered_activity[j]):
                        if j > 0:
                            j -= 1
                    adaptive_median_filtered_activity[i] = adaptive_median_filtered_activity[j]
                else:
                    adaptive_median_filtered_activity[i] = 0

            if variable_window_size:
                # Progressive increase and decrease is implemented to 
                # maintain a symmetric window around the current sam-
                # ple at the points where the distance from the cur-
                # rent sample to the end_m of the padded is less than .

                # half of the maximum window length.

                # Closer to the beginning of the signal.
                if (i < (data_length-maximum_half_window_size+minimum_half_window_size-1)):   
                    # Window grows.
                    if (half_window_size < maximum_half_window_size):
                        half_window_size += 1

                # Closer to the end of the signal.
                else: 
                    # Window shrinks.
                    if (half_window_size > minimum_half_window_size):   
                        half_window_size -= 1            

        if not self.detect_naps:
            # The improved sleep night detection is thresholding ope-
            # ration using a user-set percentile of the median activi-
            # ty.
            sleep_median_activity_threshold = np.quantile(adaptive_median_filtered_activity, self.sleep_median_activity_quantile_threshold, interpolation='linear')
            
            if sleep_median_activity_threshold < self.minimum_short_window_activity_median_threshold:
                sleep_median_activity_threshold = self.minimum_short_window_activity_median_threshold
                self.condition = 2

            improved_sleep_detection = np.where(adaptive_median_filtered_activity > sleep_median_activity_threshold, 1, 0)    # Quantile thresholding operation

            # print("invalid_zeros_mask",invalid_zeros_mask)
            # print("median_filter_half_window_size",median_filter_half_window_size)
            # print("maximum_activity",maximum_activity)
            # print("pad_size",pad_size)

        else:
            # The nap detection combines the results of a thresholding
            # operation with the activity zero proportions and another
            # with the median activity.
            
            activity_zero_proportions = zero_prop_filter(self.activity,self.nap_zero_proportion_filter_window_size)   # Zero-proportion filter

            thresholded_activity_zero_proportions = np.where(activity_zero_proportions > self.nap_zero_proportion_threshold, True, False)   # Zero-proportion thresholding operation
            thresholded_median_activity = np.where(adaptive_median_filtered_activity < self.nap_median_activity_threshold, True, False)   # Quantile thresholding operation

            # Logical combination.
            if self.compute_output_naps_with_logical_and:
                improved_sleep_detection = np.where(np.logical_and(thresholded_activity_zero_proportions,thresholded_median_activity),0,1)
            else:
                improved_sleep_detection = np.where(np.logical_or(thresholded_activity_zero_proportions,thresholded_median_activity),0,1)
        
        # New morphological structuring element.
        last_morphological_filter_structuring_element_size = 2*(self.preprocessing_morphological_filter_structuring_element_size - 1) + 1
        structuring_element = np.ones(last_morphological_filter_structuring_element_size)
        
        final_sleep_detection = improved_sleep_detection.copy()
        # Last morphological operation.
        if self.apply_last_morphological_filter:  
            final_sleep_detection = binary_opening(binary_closing(improved_sleep_detection,structuring_element),structuring_element).astype(int)

        return padded_mitigated_zeros_activity,initial_sleep_detection,morphological_filtered_initial_detection,adaptive_median_filtered_activity,activity_zero_proportions,thresholded_activity_zero_proportions,thresholded_median_activity,improved_sleep_detection,final_sleep_detection,self.condition