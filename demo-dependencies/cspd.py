# -*- coding: utf-8 -*-

# Crespo algorithm class - 01/11/2019
# Julius Andretti

import numpy as np
import time, os, inspect, sys
import cProfile, pstats, io
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import mode
from datetime import datetime, timedelta, date
from scipy.ndimage import binary_closing, binary_opening
from scipy.signal import find_peaks as peak

from cspd_functions import *
from cspd_bt_refine import CSPD_BedTime_Refiner
from cspd_gt_refine import CSPD_GetUpTime_Refiner
from crespo_algorithm import CrespoAlgorithm
from functions import *

# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)
# pd.set_option('display.max_colwidth', None)
    
class CSPD:
    def __init__(self,
                 consecutive_zeros_threshold=15,
                 awake_consecutive_zeros_threshold=2,
                 sleep_consecutive_zeros_threshold=30,
                 invalid_zeros_mitigation_percentile=0.33,
                 median_filter_window_hourly_length=8,
                 adaptive_median_filter_padding_hourly_length=1,
                 sleep_median_activity_quantile_threshold=1.0/3,
                 positive_activity_median_only_quantile_threshold=False,
                 preprocessing_morphological_filter_structuring_element_size=61,
                 detect_naps=False,
                 nap_median_activity_threshold=2,
                 compute_output_naps_with_logical_and=False,
                 nap_zero_proportion_threshold=0.5,
                 nap_zero_proportion_filter_window_size=5,
                 apply_last_morphological_filter=False,
                 short_window_activity_median_threshold_quantile=0.6,
                 minimum_short_window_activity_median_threshold=1.0,
                 condition=0,

                 nap_minimum_length=20,
                 do_peak_valley_length_filter=True,
                 median_filter_short_window=45,
                 peak_valley_minimum_length=15,
                 sleep_minimum_length=120,
                 median_filter_half_window_size=4,
                 short_window_activity_median_minimum_high_epochs=3,
                
                 refinement_maximum_allowed_datetime_gap=10,
                 sleep_maximum_allowed_datetime_gap=60,
                 bedtime_scores=[0.233528, 0.459746, 0.000522996, 0.718822, 0.558178, 0.350738,],
                 getuptime_scores=[0.831343, 0.139121, 0.117863, 0.443673, 0.339229, 0.407142,],
                 length_thresholds=[8,18,18,17],
                 candidate_thresholds=[0.210611, 0.210611, 0.715025],
                 after_candidate_window=90,
                 half_window_around_border=60,
                 activity_median_analysis_window=3,
        
                 getuptime_metric_method=1,
                 getuptime_metric_parameter=0.5,
                 getuptime_do_remove_after_long_tall_peak=True,
                 getuptime_high_probability_awake_peak_length=30,
                 getuptime_do_remove_before_long_valley=False,
                 getuptime_high_probability_sleep_valley_length=30,
                 getuptime_update_peaks_and_valleys=False,
                 getuptime_score_first_candidate=True,

                 bedtime_metric_method=1,
                 bedtime_metric_parameter=0.4,
                 bedtime_do_remove_before_long_peak=False,
                 bedtime_do_remove_before_tall_peak=False,
                 bedtime_high_probability_awake_peak_length=30,
                 bedtime_do_remove_after_long_valley=False,
                 bedtime_do_bedtime_candidates_crossings_filter=True,
                 bedtime_consider_second_best_candidate=False,
                 bedtime_update_peaks_and_valleys=False,
                 bedtime_high_probability_sleep_valley_length=30,
                 bedtime_score_last_candidate=True,
                 ):
        """Adds datetime.datetime information to a DataFrame containing the
        location of off or onwrist periods.


        Parameters
        ----------
        ### Crespo algorithm ###
        consecutive_zeros_threshold : int
                Threshold for consecutive zeroes
        awake_consecutive_zeros_threshold : int
                Threshold for consecutive zeroes inside wake periods
        sleep_consecutive_zeros_threshold : int
                Threshold for consecutive zeroes inside sleep periods
        invalid_zeros_mitigation_percentile : float
                Percentile chosen to minimize the effect of invalid zeroes 
                during preprocessing.
        median_filter_window_hourly_length : float 
                Length in hours of the preprocessing median filter.
        adaptive_median_filter_padding_hourly_length : float
                Length in hours of the padding added prior to adaptive me-
                dian filter.
        sleep_median_activity_quantile_threshold : float
                Average number of hours of sleep
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

        ### Refinement stage parameters ###
        nap_minimum_length : int
                Naps shorter than this parameter will be filtered out.
        do_peak_valley_length_filter : boolean
                If True, initial peaks or valleys (detected using the 
                Crespo algorithm) that are too short will be filtered
                out.
        peak_valley_minimum_length : int
                Initial peaks or valleys shorter than this parameter 
                may be filtered out.
        median_filter_short_window : int
                Window size used in the auxiliary median filter.
        sleep_minimum_length : int 
                Sleep nights shorter than this parameter will be filte-
                red out.
        short_window_activity_median_minimum_high_epochs : int [epochs]
                Minimum number of epochs with a "high" (above computed thre-
                shold) value of activity short-window-median to indicate the
                beggining or the end of an awake period. This is one of the 
                criteria considered when defining the borders of the refine-
                ment window.        
        refinement_maximum_allowed_datetime_gap : int [seconds]
                If the datetime seconds gap between 2 consecutive epochs is
                greater than or equal to this value, it's considered invalid.
        sleep_maximum_allowed_datetime_gap : int 
                If the datetime seconds gap between 2 consecutive epochs in-
                side a potential sleep night is greater than or equal to this 
                value, that sleep night will be divided into 2 potential sleep 
                nights to be refined.
        bedtime_scores : list [float]
                Contains all the extra scores that may be assigned to transi-
                tion candidates in the bedtime refinement context.
        getuptime_scores : list [float]
                Contains all the extra scores that may be assigned to transi-
                tion candidates in the getuptime refinement context.
        length_thresholds : list [int]
                Contains the minimum lengths for valid peaks and valleys in
                both bedtime and getuptime refinement contexts.
        candidate_thresholds : list [float]
                Contains the thresholds for criteria that will be evaluated
                for refined transition candidates.

        median_filter_half_window_size : int
                Length in minutes of the half-window to be used in the me-
                dian filter.
        after_candidate_window : int [minutes]
                Defines the above mentioned "after", a window after the can-
                didate.
        half_window_around_border : int [minutes]
                Defines a window after the current ending of the bedtime re-
                finement interval.
        activity_median_analysis_window : int [minutes]
                Used to define a median evaluation window to determine refi-
                nement interval border.

        getuptime_metric_method : {1, 2}
                If 1: the threshold will be computed as an user-set fraction 
                of the mean non-zero activity median in the refinement window.
                If 2: the threshold will be computed as an user-set quantile
                of the non-zero activity median in the refinement window.
        getuptime_metric_parameter : float
                If metric_method==1: fraction of the mean non-zero activity 
                median in the refinement window to define as metric.
                If metric_method==2: quantile of the non-zero activity median 
                in the refinement window to define as metric.
        getuptime_high_probability_sleep_valley_length : int
                If a valley with at least this length is identified, it'll
                be understood that it's highly probable that the subject is 
                asleep inside it.
        getuptime_high_probability_awake_peak_length : int
                If a peak with at least this length is identified, it'll
                be understood that it's highly probable that the subject is 
                awake inside it.
        getuptime_do_remove_after_long_tall_peak : boolean
                If True, peaks and valleys after the last identified long
                tall peak will be removed.
        getuptime_do_remove_before_long_valley : boolean
                If True, peaks and valleys before the last identified long
                valley will be removed.
        getuptime_score_first_candidate : boolean
                If True, the first candidate will get an extra score.

        bedtime_metric_method : {1, 2}
                If 1: the threshold will be computed as an user-set fraction 
                of the mean non-zero activity median in the refinement window.
                If 2: the threshold will be computed as an user-set quantile
                of the non-zero activity median in the refinement window.
        bedtime_metric_parameter : float
                If metric_method==1: fraction of the mean non-zero activity 
                median in the refinement window to define as metric.
                If metric_method==2: quantile of the non-zero activity median 
                in the refinement window to define as metric.
        bedtime_high_probability_sleep_valley_length : int
                If a valley with at least this length is identified, it'll
                be understood that it's highly probable that the subject is 
                asleep inside it.
        bedtime_bedtime_high_probability_awake_peak_length : int
                If a peak with at least this length is identified, it'll
                be understood that it's highly probable that the subject is 
                awake inside it.
        bedtime_do_remove_after_long_valley : boolean
                If True, peaks and valleys after the first identified long
                valley will be removed.
        bedtime_do_remove_before_long_peak : boolean
                If True, peaks and valleys before the last identified long
                peak will be removed.
        bedtime_do_remove_before_tall_peak : boolean
                If True, peaks and valleys before the last identified tall
                peak will be removed.
        bedtime_score_last_candidate : boolean
                If True, the last candidate will get an extra score.
        """

        self.positive_activity_median_only_quantile_threshold = positive_activity_median_only_quantile_threshold

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
        self.short_window_activity_median_threshold_quantile=short_window_activity_median_threshold_quantile
        self.minimum_short_window_activity_median_threshold=minimum_short_window_activity_median_threshold
        self.condition=condition

        self.nap_minimum_length=nap_minimum_length
        self.do_peak_valley_length_filter=do_peak_valley_length_filter
        self.median_filter_short_window=median_filter_short_window
        self.peak_valley_minimum_length=peak_valley_minimum_length
        self.sleep_minimum_length=sleep_minimum_length
        self.short_window_activity_median_minimum_high_epochs=short_window_activity_median_minimum_high_epochs
                
        self.refinement_maximum_allowed_datetime_gap=refinement_maximum_allowed_datetime_gap
        self.sleep_maximum_allowed_datetime_gap=sleep_maximum_allowed_datetime_gap
        self.bedtime_scores=bedtime_scores
        self.getuptime_scores=getuptime_scores
        self.length_thresholds=length_thresholds
        self.candidate_thresholds=candidate_thresholds

        self.median_filter_half_window_size=median_filter_half_window_size
        self.after_candidate_window=after_candidate_window
        self.half_window_around_border=half_window_around_border
        self.activity_median_analysis_window=activity_median_analysis_window
        
        self.getuptime_metric_method=getuptime_metric_method
        self.getuptime_metric_parameter=getuptime_metric_parameter
        self.getuptime_do_remove_after_long_tall_peak=getuptime_do_remove_after_long_tall_peak
        self.getuptime_high_probability_awake_peak_length=getuptime_high_probability_awake_peak_length
        self.getuptime_do_remove_before_long_valley=getuptime_do_remove_before_long_valley
        self.getuptime_high_probability_sleep_valley_length=getuptime_high_probability_sleep_valley_length
        self.getuptime_update_peaks_and_valleys=getuptime_update_peaks_and_valleys
        self.getuptime_score_first_candidate=getuptime_score_first_candidate

        self.bedtime_metric_method=bedtime_metric_method
        self.bedtime_metric_parameter=bedtime_metric_parameter
        self.bedtime_do_remove_before_long_peak=bedtime_do_remove_before_long_peak
        self.bedtime_do_remove_before_tall_peak=bedtime_do_remove_before_tall_peak
        self.bedtime_update_peaks_and_valleys=bedtime_update_peaks_and_valleys
        self.bedtime_do_bedtime_candidates_crossings_filter=bedtime_do_bedtime_candidates_crossings_filter
        self.bedtime_consider_second_best_candidate = bedtime_consider_second_best_candidate
        self.bedtime_high_probability_awake_peak_length=bedtime_high_probability_awake_peak_length
        self.bedtime_do_remove_after_long_valley=bedtime_do_remove_after_long_valley
        self.bedtime_high_probability_sleep_valley_length=bedtime_high_probability_sleep_valley_length
        self.bedtime_score_last_candidate=bedtime_score_last_candidate

    
    def model(self,
              activity,
              datetime_stamps,
              verbose=False,
              ):
        """Adds datetime.datetime information to a DataFrame containing the
        location of off or onwrist periods.

        Parameters
        ----------
        activity : np.array [float]
                input activity array
        datetime_stamps : np.array [datetime.datetime]
                datetime array containing the actigraphy timestamps
        verbose : boolean
                Verbosity
        """
        self.datetime_stamps = datetime_stamps
        self.activity = activity


        # An input is valid if its epoch duration is not greater than 120s 
        # and its total duration is at least a hour.
        self.invalid_input = False


        data_length = len(datetime_stamps)
        if data_length > 1:
            if not isinstance(datetime_stamps[0],datetime):
                if isinstance(datetime_stamps[0],str):
                    datetime_stamps = pd.to_datetime(datetime_stamps,dayfirst=True)
                
                datetime_stamps = np.array([to_datetime(stamp) for stamp in datetime_stamps])

            self.duration = mode(datetime_diff(datetime_stamps),keepdims=True).mode[0]   # Epoch duration, most repeated interval between measures
            self.datetime_stamps = datetime_stamps

            if (
                (self.duration > 120)
                or 
                (((self.datetime_stamps[data_length-1] - self.datetime_stamps[0]).total_seconds()) < 3600)
                ):
                self.invalid_input = True

            else:
                self.activity = (1.0/self.duration)*np.array(activity)   # Activity input is scaled
                
                if self.positive_activity_median_only_quantile_threshold:
                    zp = zero_prop(activity)
                    sleep_median_activity_quantile_threshold = zp + self.sleep_median_activity_quantile_threshold*(1-zp)
                else:
                    sleep_median_activity_quantile_threshold = self.sleep_median_activity_quantile_threshold
                # if sleep_median_activity_quantile_threshold > max_sq:
                #     sleep_median_activity_quantile_threshold = max_sq
                    
        else:
            self.invalid_input = True
            self.duration = 0

        if not self.invalid_input:
            epoch_hour = 3600/self.duration

            crespo = CrespoAlgorithm(self.consecutive_zeros_threshold,
                                     self.awake_consecutive_zeros_threshold,
                                     self.sleep_consecutive_zeros_threshold,
                                     self.invalid_zeros_mitigation_percentile,
                                     self.median_filter_window_hourly_length,
                                     self.adaptive_median_filter_padding_hourly_length,
                                     sleep_median_activity_quantile_threshold,
                                     self.preprocessing_morphological_filter_structuring_element_size,
                                     self.detect_naps,
                                     self.nap_median_activity_threshold,
                                     self.compute_output_naps_with_logical_and,
                                     self.nap_zero_proportion_threshold,
                                     self.nap_zero_proportion_filter_window_size,
                                     self.apply_last_morphological_filter,
                                     self.minimum_short_window_activity_median_threshold,
                                     self.condition
                                     )
            padded_mitigated_zeros_activity,initial_sleep_detection,morphological_filtered_initial_detection,adaptive_median_filtered_activity,activity_zero_proportions,thresholded_activity_zero_proportions,thresholded_median_activity,improved_sleep_detection,final_sleep_detection,condition = crespo.detect_msp(self.activity,
                                                                                                                                                                                                                                                                                                                        self.datetime_stamps,
                                                                                                                                                                                                                                                                                                                        epoch_hour,
                                                                                                                                                                                                                                                                                                                        )

            self.activity_zero_proportions = activity_zero_proportions
            self.thresholded_activity_zero_proportions = thresholded_activity_zero_proportions
            self.thresholded_median_activity = thresholded_median_activity
            self.condition = condition


            bedtime_last_candidate_score,bedtime_best_median_difference_candidate_score,bedtime_best_crossing_distance_candidate_score,bedtime_best_epochs_above_metric_after_score,bedtime_thresholded_candidate_score_amplitude,bedtime_thresholded_candidate_score_minimum = self.bedtime_scores

            getuptime_first_candidate_score,getuptime_best_median_difference_candidate_score,getuptime_best_crossing_distance_candidate_score,getuptime_best_epochs_above_metric_after_score,getuptime_thresholded_candidate_score_amplitude,getuptime_thresholded_candidate_score_minimum = self.getuptime_scores

            bedtime_minimum_peak_length,bedtime_minimum_valley_length,getuptime_minimum_peak_length,getuptime_minimum_valley_length = np.array(self.length_thresholds).astype(int)

            bedtime_maximum_epochs_above_metric_after_candidate,getuptime_maximum_epochs_above_metric_after_candidate,zero_proportion_threshold = self.candidate_thresholds

            if self.condition == 2:
                zero_proportion_threshold *= 0.666
                short_window_activity_median_threshold_quantile = 1.333*self.short_window_activity_median_threshold_quantile
            else:
                short_window_activity_median_threshold_quantile = self.short_window_activity_median_threshold_quantile

            refinement_maximum_allowed_datetime_gap = self.refinement_maximum_allowed_datetime_gap*60
            sleep_maximum_allowed_datetime_gap = self.sleep_maximum_allowed_datetime_gap*60

            # Minutes to epochs conversion
            bedtime_minimum_peak_length = int(round(bedtime_minimum_peak_length*60/self.duration))
            bedtime_minimum_valley_length = int(round(bedtime_minimum_valley_length*60/self.duration))
            getuptime_minimum_peak_length = int(round(getuptime_minimum_peak_length*60/self.duration))
            getuptime_minimum_valley_length = int(round(getuptime_minimum_valley_length*60/self.duration))
            after_candidate_window = int(round(self.after_candidate_window*60/self.duration))
            bedtime_maximum_epochs_above_metric_after_candidate = int(round(self.after_candidate_window*bedtime_maximum_epochs_above_metric_after_candidate))
            getuptime_maximum_epochs_above_metric_after_candidate = int(round(self.after_candidate_window*getuptime_maximum_epochs_above_metric_after_candidate))
            half_window_around_border = int(round(self.half_window_around_border*60/self.duration))
            activity_median_analysis_window = int(round(self.activity_median_analysis_window*60/self.duration))
            median_filter_half_window_size = int(round(self.median_filter_half_window_size*60/self.duration))
            peak_valley_minimum_length = int(round(self.peak_valley_minimum_length*60/self.duration))
            median_filter_short_window = int(self.median_filter_short_window*60/self.duration)
            short_window_activity_median_minimum_high_epochs = int(self.short_window_activity_median_minimum_high_epochs*60/self.duration)

            if verbose:
                print("sleep_median_activity_quantile_threshold ",self.sleep_median_activity_quantile_threshold)

            # sleep_activity = self.ac

            # Auxiliary small-window median filter
            short_window_activity_median = median_filter(self.activity,median_filter_short_window)
            short_window_activity_median_threshold = np.quantile(short_window_activity_median, short_window_activity_median_threshold_quantile, interpolation='linear')

            # Sequences of zeros are filtered if they aren't long enough
            if self.do_peak_valley_length_filter:
                initial_peaks_and_valleys = identify_peaks_and_valleys(self.activity,final_sleep_detection,0.5)

                peaks_and_valleys_count = len(initial_peaks_and_valleys)

                if (initial_peaks_and_valleys.at[0,"length"] <= peak_valley_minimum_length):
                    start = int(initial_peaks_and_valleys.at[0,"start"])
                    end = int(initial_peaks_and_valleys.at[1,"end"])

                    initial_peaks_and_valleys.at[1,"start"] = start

                    length = end-start
                    initial_peaks_and_valleys.at[1,"length"] = length

                    initial_peaks_and_valleys.drop(index=[0],inplace=True)
                    peaks_and_valleys_count = len(initial_peaks_and_valleys)
                    initial_peaks_and_valleys.index = range(peaks_and_valleys_count)

                region_index = 1
                while (region_index < peaks_and_valleys_count):
                    remove = False
                    # All transitions are subjected to length-based thresholding and, 
                    # if not filtered, they are evaluated based off other criteria.

                    if (initial_peaks_and_valleys.at[region_index,"length"] <= peak_valley_minimum_length):
                        remove = True

                    if remove:
                        # Transition filtering process takes place. When a valley or 
                        # a peak is removed, it's actually merged into the neighbor-
                        # ing regions to create a large new region composed by the 3
                        # (or 2, if it's the penultimate) regions.

                        start = int(initial_peaks_and_valleys.at[region_index-1,"start"])

                        if region_index < peaks_and_valleys_count-1:
                            end = int(initial_peaks_and_valleys.at[region_index+1,"end"])
                            initial_peaks_and_valleys.drop(index=[region_index,region_index+1],inplace=True)

                        else:
                            end = int(initial_peaks_and_valleys.at[region_index,"end"])
                            initial_peaks_and_valleys.drop(index=[region_index],inplace=True)

                        initial_peaks_and_valleys.at[region_index-1,"end"] = end

                        length = end-start
                        initial_peaks_and_valleys.at[region_index-1,"length"] = length

                        peaks_and_valleys_count = len(initial_peaks_and_valleys)
                        initial_peaks_and_valleys.index = range(peaks_and_valleys_count)

                    else:
                        region_index += 1

                final_sleep_detection = np.ones(data_length)
                peaks_and_valleys_count = len(initial_peaks_and_valleys)
                region_index = 0
                while (region_index < peaks_and_valleys_count):
                    if (initial_peaks_and_valleys.at[region_index,"class"] == "v"):
                        final_sleep_detection[int(initial_peaks_and_valleys.at[region_index,"start"]):int(initial_peaks_and_valleys.at[region_index,"end"])] = 0
                    region_index += 1


            datetime_difference = datetime_diff(self.datetime_stamps)
            # Index of "big" gaps within possible sleep periods.
            sleep_gaps = [l for l in range(data_length) if ((datetime_difference[l] > sleep_maximum_allowed_datetime_gap) and (final_sleep_detection[l] == 0))]

            # Big gaps within sleep periods indicate a probable merge of two sleep pe-
            # riods, so we separate them.
            for l in sleep_gaps: 
                final_sleep_detection[l-10:l+11] = 1

            sleep_period_borders = np.diff(final_sleep_detection)

            # Output sleep period borders.
            sleep_period_borders_index = sleep_period_borders.nonzero()[0]
            if final_sleep_detection[0] == 0:
                sleep_period_borders[0] = -1.0
                sleep_period_borders_index = np.hstack(([0],sleep_period_borders_index))

            # The first epoch of each transition is stored with their direction (+1 for
            # sleep to awake).
            transitions = [[i,sleep_period_borders[i]] for i in sleep_period_borders_index]
            refined_transitions = transitions.copy()
            num_transitions = len(refined_transitions)

            if verbose:
                print("pre-refinement transitions")
                for transition in transitions:
                    if transition[1] > 0:
                        print("getup time: ",self.datetime_stamps[transition[0]])
                    else:
                        print("bed time: ",self.datetime_stamps[transition[0]])

            if self.condition == 1:
                median_excursion_threshold *= 3.5

            refinement_window_median = np.zeros(data_length)
            refinement_window_levels = np.zeros(data_length)
            refinement_window_metric_threshold = np.zeros(data_length)
            refinement_window_median_difference = np.zeros(data_length)

            # Refined bed times indexes array.
            refined_bedtimes = np.array([],dtype=int)
            
            # Refined wakeup times indexes array.
            refined_getuptimes = np.array([],dtype=int)

            # Transitions are refined sequentially.
            window_metrics = []
            if num_transitions > 0: 
                window_metrics = np.zeros(num_transitions)

                refined_output = np.zeros(data_length)

                bedtime_refiner = CSPD_BedTime_Refiner(self.activity,self.datetime_stamps,
                                                       short_window_activity_median,
                                                       self.minimum_short_window_activity_median_threshold,

                                                       short_window_activity_median_minimum_high_epochs,
                                                       half_window_around_border,
                                                       activity_median_analysis_window,
                                                       refinement_maximum_allowed_datetime_gap,
                                                       short_window_activity_median_threshold,
                                                       median_filter_half_window_size,
                                                       self.bedtime_metric_method,
                                                       self.bedtime_metric_parameter,
                                                       self.condition,

                                                       self.bedtime_score_last_candidate,
                                                       bedtime_last_candidate_score,
                                                       bedtime_best_median_difference_candidate_score,
                                                       bedtime_best_crossing_distance_candidate_score,
                                                       bedtime_best_epochs_above_metric_after_score,
                                                       bedtime_thresholded_candidate_score_amplitude,
                                                       bedtime_thresholded_candidate_score_minimum,
                                                       bedtime_minimum_peak_length,
                                                       bedtime_minimum_valley_length,

                                                       after_candidate_window,
                                                       bedtime_maximum_epochs_above_metric_after_candidate,
                                                       zero_proportion_threshold,

                                                       self.bedtime_do_remove_after_long_valley,
                                                       self.bedtime_do_remove_before_long_peak,
                                                       self.bedtime_do_remove_before_tall_peak,
                                                       self.bedtime_update_peaks_and_valleys,
                                                       self.bedtime_do_bedtime_candidates_crossings_filter,
                                                       self.bedtime_consider_second_best_candidate,
                                                       self.bedtime_high_probability_awake_peak_length,
                                                       self.bedtime_high_probability_sleep_valley_length,
                                                       )

                getuptime_refiner = CSPD_GetUpTime_Refiner(self.activity,self.datetime_stamps,
                                                           short_window_activity_median,
                                                           self.minimum_short_window_activity_median_threshold,
                                                           
                                                           short_window_activity_median_minimum_high_epochs,
                                                           half_window_around_border,
                                                           activity_median_analysis_window,
                                                           refinement_maximum_allowed_datetime_gap,
                                                           short_window_activity_median_threshold,
                                                           median_filter_half_window_size,
                                                           self.getuptime_metric_method,
                                                           self.getuptime_metric_parameter,
                                                           self.condition,

                                                           getuptime_first_candidate_score,
                                                           getuptime_best_median_difference_candidate_score,
                                                           getuptime_best_crossing_distance_candidate_score,
                                                           getuptime_best_epochs_above_metric_after_score,
                                                           getuptime_thresholded_candidate_score_amplitude,
                                                           getuptime_thresholded_candidate_score_minimum,
                                                           getuptime_minimum_peak_length,
                                                           getuptime_minimum_valley_length,
                                                           after_candidate_window,
                                                           getuptime_maximum_epochs_above_metric_after_candidate,
                                                           zero_proportion_threshold,

                                                           self.getuptime_do_remove_after_long_tall_peak,
                                                           self.getuptime_high_probability_awake_peak_length,
                                                           self.getuptime_high_probability_sleep_valley_length,
                                                           self.getuptime_do_remove_before_long_valley,
                                                           self.getuptime_update_peaks_and_valleys,
                                                           self.getuptime_score_first_candidate,
                                                           )

                i = 0
                while i < num_transitions:
                    initial_transition_candidate = transitions[i][0]

                    if i-1 >= 0:
                        previous_transition = refined_transitions[i-1][0]
                    else:
                        previous_transition = 0

                    if i+1 < num_transitions:
                        next_transition = refined_transitions[i+1][0]
                    else:
                        next_transition = data_length+1

                    if transitions[i][1] < 0:   # Bedtime refinement.
                        refined_bedtime,refinement_window_start,refinement_window_end,refinement_window_activity_median,refinement_window_activity_median_difference_smoothed,refinement_window_levels,refinement_window_metric = bedtime_refiner.refine(refinement_window_levels,
                                                                                                                                                                                                                                                         initial_transition_candidate,
                                                                                                                                                                                                                                                         previous_transition,
                                                                                                                                                                                                                                                         next_transition,
                                                                                                                                                                                                                                                         verbose=verbose
                                                                                                                                                                                                                                                         )

                        refined_bedtimes = np.append(refined_bedtimes,refined_bedtime)
                        refined_transitions[i][0] = refined_bedtime

                        refined_output[previous_transition:refined_bedtime] = 1

                    else:   # Getup time refinement.
                        refined_getuptime,refinement_window_start,refinement_window_end,refinement_window_activity_median,refinement_window_activity_median_difference_smoothed,refinement_window_levels,refinement_window_metric = getuptime_refiner.refine(refinement_window_levels,
                                                                                                                                                                                                                                                             initial_transition_candidate,
                                                                                                                                                                                                                                                             previous_transition,
                                                                                                                                                                                                                                                             next_transition,
                                                                                                                                                                                                                                                             verbose=verbose
                                                                                                                                                                                                                                                             )


                        refined_getuptimes = np.append(refined_getuptimes,refined_getuptime)
                        refined_transitions[i][0] = refined_getuptime

                    if refinement_window_end > refinement_window_start:
                        refinement_window_median[refinement_window_start:refinement_window_end+1] = refinement_window_activity_median
                        refinement_window_median_difference[refinement_window_start+1:refinement_window_end+1] = refinement_window_activity_median_difference_smoothed
                        refinement_window_metric_threshold[refinement_window_start:refinement_window_end+1] = refinement_window_metric

                    window_metrics[i] = refinement_window_metric

                    i += 1

                if refined_transitions[-1][1] > 0:
                    refined_output[refined_transitions[-1][0]:data_length] = 1

                if verbose:
                    print("window_metrics")
                    print(window_metrics)
                    print("mean_window_metrics",np.mean(window_metrics))

            else:
                refined_output = np.ones(data_length)

    
            self.nap_minimum_length = int(round(self.nap_minimum_length*60/self.duration))
            
            if self.detect_naps:
                final_sleep_detection = boolean_length_filter(self.nap_minimum_length,refined_output)

                final_sleep_detection = peak_valley_zero_proportion_filter(1/3,final_sleep_detection)

                final_sleep_detection = boolean_length_filter(int(0.5*self.nap_minimum_length),final_sleep_detection,class_to_filter="p")

                refined_output = final_sleep_detection.copy()
                
            else:
                final_sleep_detection = boolean_length_filter(self.sleep_minimum_length,refined_output)
            
            refined_output = final_sleep_detection.copy()
            sleep_period_borders = np.diff(refined_output)
            sleep_period_borders_index = sleep_period_borders.nonzero()[0]
            refined_transitions = [[i,sleep_period_borders[i]] for i in sleep_period_borders_index]

            
            if len(refined_transitions) > 0:
                self.refined_transitions = np.array([[self.datetime_stamps[rt[0]],*rt] for rt in refined_transitions])

                self.refined_sleep_df = pd.DataFrame([[self.refined_transitions[2*i][0],self.refined_transitions[2*i+1][0],self.refined_transitions[2*i][1],self.refined_transitions[2*i+1][1]] for i in range(len(self.refined_transitions)//2)],columns=["bedtime","getuptime","bedtime_index","getuptime_index"])

                timedeltas = (self.refined_sleep_df["getuptime"]-self.refined_sleep_df["bedtime"]).values
                self.refined_sleep_df["hour_length"] = [timedelta/np.timedelta64(1,'s')/3600 for timedelta in timedeltas]

            else:
                self.refined_transitions = np.array([[pd.Timestamp(date.today()+timedelta(days=3*365)),0,0]])
                self.refined_sleep_df = pd.DataFrame([])

            # Heuristic rule from Crespo algorithm.
            final_sleep_detection[0] = 1
            final_sleep_detection[-1] = 1
            refined_output[0] = 1
            refined_output[-1] = 1

            # self.sleep_median_activity_quantile_threshold = self.duration*np.quantile(adaptive_median_filtered_activity, self.sleep_median_activity_quantile_threshold, interpolation='linear')
            self.refined_bed_times = refined_bedtimes
            self.refined_getup_times = refined_getuptimes

        else:
            padded_mitigated_zeros_activity = None
            short_window_activity_median = None
            initial_sleep_detection = None
            morphological_filtered_initial_detection = None
            adaptive_median_filtered_activity = None
            improved_sleep_detection = None
            self.short_window_activity_median_threshold = 0
            refinement_window_median = None
            refinement_window_levels = None
            refinement_window_metric_threshold = None
            refinement_window_median_difference = None
            refined_bed_times = None
            refined_getup_times = None
            self.sleep_median_activity_quantile_threshold = 0
            
            final_sleep_detection = np.ones(data_length)
            refined_output = np.ones(data_length)
            self.refined_transitions = []
            self.refined_bed_times = []
            self.refined_getup_times = []

        if not self.detect_naps:
            self.padded_mitigated_zeros_activity = padded_mitigated_zeros_activity
            self.initial_sleep_detection = initial_sleep_detection
            self.morphological_filtered_initial_detection = morphological_filtered_initial_detection
            if not self.invalid_input:
                self.adaptive_median_filtered_activity = self.duration*adaptive_median_filtered_activity
            else:
                self.adaptive_median_filtered_activity = adaptive_median_filtered_activity

        self.window_metrics = window_metrics
        self.improved_sleep_detection = improved_sleep_detection
        self.short_window_activity_median = short_window_activity_median
        self.refinement_window_median = refinement_window_median
        self.refinement_window_levels = refinement_window_levels
        self.refinement_window_metric_threshold = refinement_window_metric_threshold
        self.refinement_window_median_difference = refinement_window_median_difference
        self.final_sleep_detection = final_sleep_detection
        self.refined_output = refined_output