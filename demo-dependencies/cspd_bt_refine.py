"""Bedtime refinement class for the CPSD algorithm

Author: Julius A. P. P. de Paula (--/2023)
"""

import os, inspect, sys
os.environ['NUMPY_EXPERIMENTAL_ARRAY_FUNCTION'] = '0'
import cProfile, pstats, io
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
from scipy.signal import find_peaks as peak


from cspd_functions import *
from functions import *


class CSPD_BedTime_Refiner:
    def __init__(self,
                 activity,
                 datetime_stamps,
                 short_window_activity_median,
                 minimum_short_window_activity_median_threshold,

                 short_window_activity_median_minimum_high_epochs,
                 half_window_around_border,
                 activity_median_analysis_window,
                 maximum_allowed_gap,
                 quantile_threshold,
                 median_filter_half_window_size,
                 metric_method,
                 metric_parameter,
                 condition,

                 bedtime_score_last_candidate,
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

                 do_remove_after_long_valley,
                 do_remove_before_long_peak,
                 do_remove_before_tall_peak,
                 update_peaks_and_valleys,
                 do_bedtime_candidates_crossings_filter,
                 consider_second_best_candidate,
                 bedtime_high_probability_awake_peak_length,
                 bedtime_high_probability_sleep_valley_length,
               ):
        """Searches for the epoch that shows the best features to represent
        the transition from awake (out-of-bed) to sleep (in-bed) state, the
        "bed time".

        Parameters
        ----------
        activity : np.array [float]
                Read activity information.
        datetime_stamps : np.array [datetime stamp]
                datetime.datetime stamps read by the device.
        short_window_activity_median : np.array [float]
                Median activity computed from read activity using a short
                filter window. Captures higher frequency variations in the
                input activity.

        maximum_allowed_gap : int [seconds]
                If the datetime seconds gap between 2 consecutive epochs is
                greater than or equal to this value, it's considered invalid.
        median_filter_half_window_size : int
                Defines the size of the window used when computing the median
                filter inside the refinement window.
        half_window_around_border : int [epochs]
                The proportion of zeros around the border of a refinement win-
                dow will be computed using a window of size 2*half_window_a-
                round_border+1 centered in that border.
        zero_proportion_threshold : float
                Minimum proportion of zeros for a valid refinement window end
                candidate.
        metric_method : {1, 2}
                If 1: the threshold will be computed as an user-set fraction 
                of the mean non-zero activity median in the refinement window.
                If 2: the threshold will be computed as an user-set quantile
                of the non-zero activity median in the refinement window.
        metric_parameter : float
                If metric_method==1: fraction of the mean non-zero activity 
                median in the refinement window to define as metric.
                If metric_method==2: quantile of the non-zero activity median 
                in the refinement window to define as metric.
        short_window_activity_median_minimum_high_epochs : int [epochs]
                Minimum number of epochs with a "high" (above computed thre-
                shold) value of activity short-window-median to indicate the
                beggining or the end of an awake period. This is one of the 
                criteria considered when defining the borders of the refine-
                ment window.
        activity_median_analysis_window : int [epochs]
                Minimum number of epochs with a "low" (below computed thre-
                shold) value of activity short-window-median to indicate the
                beggining or the end of a sleep period. This is one of the 
                criteria considered when defining the borders of the refine-
                ment window.
        quantile_threshold : float
                Quantile of the short-window-median-filtered activity used
                as a threshold of separation between low and high levels.
        condition : int
                User-set configuration. Used to indicate if the subject suf-
                fers from a particular condition.

        bedtime_high_probability_sleep_valley_length : int
                If a valley with at least this length is identified, it'll
                be understood that it's highly probable that the subject is 
                asleep inside it.
        bedtime_high_probability_awake_peak_length : int
                If a peak with at least this length is identified, it'll
                be understood that it's highly probable that the subject is 
                awake inside it.
        do_remove_after_long_valley : boolean
                If True, peaks and valleys after the first identified long
                valley will be removed.
        do_remove_before_long_peak : boolean
                If True, peaks and valleys before the last identified long
                peak will be removed.
        do_remove_before_tall_peak : boolean
                If True, peaks and valleys before the last identified tall
                peak will be removed.

        bedtime_minimum_valley_length : int
                Valleys shorter than this parameter may be removed.
        bedtime_minimum_peak_length : int
                Peaks shorter than this parameter may be removed.

        bedtime_score_last_candidate : boolean
                If True, the last candidate will get an extra score.
        bedtime_last_candidate_score : float
                Extra score for the last candidate.
        bedtime_best_median_difference_candidate_score : float
                Extra score for the candidate nearest to the sharpest drop
                in median activity.
        bedtime_best_crossing_distance_candidate_score : float
                Extra score for the candidate nearest to the last short-
                window-median-filtered activity metric crossing-down.

        after_candidate_window : int
                Length of the window of epochs after the candidate that
                will be analyzed to determine how many are above the me-
                tric threshold.
        bedtime_maximum_epochs_above_metric_after_candidate : int
                Candidates with fewer epoch above metric after them than
                this parameter ("thresholded") will get extra scores.
        bedtime_best_epochs_above_metric_after_score : float
                If there aren't any candidates with few enough epochs a-
                bove metric following, the one with less gets this extra
                score.
        bedtime_thresholded_candidate_score_amplitude : float
                Maximum (before adding the minimum score) extra score for 
                a "thresholded" candidate.
        bedtime_thresholded_candidate_score_minimum : float
                Minimum extra score for a "thresholded" candidate.
        """

        self.activity = activity
        self.datetime_stamps = datetime_stamps
        self.short_window_activity_median = short_window_activity_median
        self.minimum_short_window_activity_median_threshold = minimum_short_window_activity_median_threshold

        self.short_window_activity_median_minimum_high_epochs = short_window_activity_median_minimum_high_epochs
        self.half_window_around_border = half_window_around_border
        self.activity_median_analysis_window = activity_median_analysis_window
        self.maximum_allowed_gap = maximum_allowed_gap
        self.quantile_threshold = quantile_threshold
        self.median_filter_half_window_size = median_filter_half_window_size
        self.metric_method = metric_method
        self.metric_parameter = metric_parameter
        self.condition = condition

        self.bedtime_score_last_candidate = bedtime_score_last_candidate
        self.bedtime_last_candidate_score = bedtime_last_candidate_score
        self.bedtime_best_median_difference_candidate_score = bedtime_best_median_difference_candidate_score
        self.bedtime_best_crossing_distance_candidate_score = bedtime_best_crossing_distance_candidate_score
        self.bedtime_best_epochs_above_metric_after_score = bedtime_best_epochs_above_metric_after_score
        self.bedtime_thresholded_candidate_score_amplitude = bedtime_thresholded_candidate_score_amplitude
        self.bedtime_thresholded_candidate_score_minimum = bedtime_thresholded_candidate_score_minimum
        self.bedtime_minimum_peak_length = bedtime_minimum_peak_length
        self.bedtime_minimum_valley_length = bedtime_minimum_valley_length

        self.after_candidate_window = after_candidate_window
        self.bedtime_maximum_epochs_above_metric_after_candidate = bedtime_maximum_epochs_above_metric_after_candidate
        self.zero_proportion_threshold = zero_proportion_threshold

        self.do_remove_after_long_valley = do_remove_after_long_valley
        self.do_remove_before_long_peak = do_remove_before_long_peak
        self.do_remove_before_tall_peak = do_remove_before_tall_peak
        self.update_peaks_and_valleys = update_peaks_and_valleys
        self.do_bedtime_candidates_crossings_filter = do_bedtime_candidates_crossings_filter
        self.consider_second_best_candidate = consider_second_best_candidate
        self.bedtime_high_probability_awake_peak_length = bedtime_high_probability_awake_peak_length
        self.bedtime_high_probability_sleep_valley_length = bedtime_high_probability_sleep_valley_length


        self.data_length = len(self.activity)


    def datetime_gap_check(self,
                           location,
                           direction="backward",
                           return_gap=False,
                           ):
        """Checks if the datetime gap from a specific epoch to the previous
        or the next has a valid length.

        Parameters
        ----------
        location : int
                Index of a specific epoch
        direction : {"backward","forward"}, default "backward".
                If "backward", the gap is computed from 'location' to the pre-
                vious epoch. If "forward", from 'location' to the next epoch.
        return_gap: boolean, default False
                If True, the computed gap is returned.

        Returns
        -------
        valid_gap : boolean
                If True, the datetime gap is valid.
        """

        if direction == "backward":
            datetime_seconds_gap = (self.datetime_stamps[location]-self.datetime_stamps[location-1]).total_seconds()
        else:
            datetime_seconds_gap = (self.datetime_stamps[location+1]-self.datetime_stamps[location]).total_seconds()

        valid_gap = True
        if datetime_seconds_gap > self.maximum_allowed_gap:
            valid_gap = False

        if return_gap:
            return valid_gap, datetime_seconds_gap

        return valid_gap

    def compute_refinement_window_median(self,
                                         refinement_window_start,
                                         refinement_window_end,
                                         ):
        """ Applies a median filter to the activity in a refinement window.

        Parameters
        ----------
        refinement_window_start : int
                Index of the first epoch of a refinement window.
        refinement_window_end : int
                Index of the last epoch of a refinement window.

        Returns
        -------
        refinement_window_activity_median : np.array [float]
                Median-filtered activity in the refinement window.
        """

        # If there aren't enough epochs available to fill the filter window,
        # the maximum value is concatenated as needed.
        if refinement_window_start-self.median_filter_half_window_size >= 0:
            if refinement_window_end+self.median_filter_half_window_size+1 <= self.data_length:
                refinement_window_activity = self.activity[refinement_window_start-self.median_filter_half_window_size:refinement_window_end+self.median_filter_half_window_size+1]
            else:
                refinement_window_activity = self.activity[refinement_window_start-self.median_filter_half_window_size:self.data_length]
                while len(refinement_window_activity) < (refinement_window_end+1+2*self.median_filter_half_window_size-refinement_window_start):
                    refinement_window_activity = np.append(refinement_window_activity,np.max(self.activity[refinement_window_start:refinement_window_end+1]))
        else:
            refinement_window_activity = self.activity[0:refinement_window_end+self.median_filter_half_window_size+1]
            while len(refinement_window_activity) < (refinement_window_end+1+2*self.median_filter_half_window_size-refinement_window_start):
                refinement_window_activity = np.insert(refinement_window_activity, 0, np.max(self.activity[refinement_window_start:refinement_window_end+1]))

        refinement_window_activity_median = median_filter(refinement_window_activity,self.median_filter_half_window_size,padding='padded')

        return refinement_window_activity_median

    def compute_zero_proportion_around_end(self,
                                           refinement_window_end,
                                           ):
        """Computes the proportion of zeros in a window centered around the 
        last epoch of a refinement window.

        Parameters
        ----------
        refinement_window_end : int
                Index of the last epoch of a refinement window.

        Returns
        -------
        zero_proportion_around_end : float
                The computed proportion of zeros.
        """

        if ((refinement_window_end+self.half_window_around_border) < self.data_length):
            window_around_border_end = refinement_window_end+self.half_window_around_border+1
        else:
            window_around_border_end = self.data_length

        if ((refinement_window_end-self.half_window_around_border) > 0):
            window_around_border_start = refinement_window_end-self.half_window_around_border
        else:
            window_around_border_start = 0

        activity_around_end = self.activity[window_around_border_start:window_around_border_end]
        zero_proportion_around_end = zero_prop(activity_around_end)

        return zero_proportion_around_end

    def compute_metric(self,
                       refinement_window_activity_median,
                       ):
        """Computes a threshold that separates low and high activity inside
        a refinement window.

        Parameters
        ----------
        refinement_window_activity_median : np.array [float]
                Median-filtered activity in the refinement window.

        Returns
        -------
        metric : float
                Threshold of separation between high and low activity in the
                context of a refinement window.
        """

        metric = 0

        # Only nonzero values will be considered when computing the threshold.
        refinement_window_positive_activity_median = refinement_window_activity_median[np.where(refinement_window_activity_median > 0)]
        if len(refinement_window_positive_activity_median) > 0:
            if self.metric_method == 1:
                mean = np.mean(refinement_window_positive_activity_median) # Mean of non-zero elements in filtered array            
                fraction_mean = self.metric_parameter*mean # Fraction of the mean median activity
                metric = fraction_mean
            elif self.metric_method == 2:
                p = np.quantile(refinement_window_positive_activity_median, self.metric_parameter, interpolation='linear') # Quantile of the median activity
                metric = p

        return metric


    def compute_initial_refinement_window_start(self,
                                                initial_candidate,
                                                ):
        """Searches for the best candidate to be the first epoch in the re-
        finement window. In this context, the best candidate is the first
        epoch starting from the initial one to the previous ones that has a
        high probability of being in the awake state.

        Parameters
        ----------
        initial_candidate : int
                Location of the initial candidate to be the transition betwe-
                en from the awake to the sleep state.

        Returns
        -------
        refinement_window_start : int
                Index of the first epoch of a refinement window.
        """

        refinement_window_start = initial_candidate
        if refinement_window_start < 0:
            refinement_window_start = 0

        # The initial refinement window start will be defined based on 2
        # criteria. First: valid datetime gaps, if an invalid gap is found
        # the epoch right after it is chosen, because the existence of a
        # gap is considered impossible when the subject is asleep.
        valid_gap = self.datetime_gap_check(refinement_window_start)

        # Second: if gaps remain valid, the search continues until an e-
        # poch with a high enough level of activity before it is found.
        if refinement_window_start >= self.short_window_activity_median_minimum_high_epochs:
            short_window_activity_median_before = self.short_window_activity_median[refinement_window_start-self.short_window_activity_median_minimum_high_epochs:refinement_window_start] 
        else:
            short_window_activity_median_before = self.short_window_activity_median[0:refinement_window_start] 

        high_short_window_activity_median_before_proportion = 1-below_prop(short_window_activity_median_before,self.quantile_threshold)

        while (   # Boundary conditions for the beggining of the bedtime refinement interval
                   (refinement_window_start > 0) and
                   (refinement_window_start-1 > self.previous_transition) and 
                   valid_gap and
                   (high_short_window_activity_median_before_proportion < 1)
              ):
            refinement_window_start -= 1   # Interval is stretched to the left

            if refinement_window_start > 0:
                valid_gap = self.datetime_gap_check(refinement_window_start)

            if refinement_window_start >= self.short_window_activity_median_minimum_high_epochs:
                short_window_activity_median_before = self.short_window_activity_median[refinement_window_start-self.short_window_activity_median_minimum_high_epochs:refinement_window_start] 
            else:
                short_window_activity_median_before = self.short_window_activity_median[0:refinement_window_start] 

            high_short_window_activity_median_before_proportion = 1-below_prop(short_window_activity_median_before,self.quantile_threshold)

        return refinement_window_start

    def compute_refinement_window_end(self,
                                      initial_candidate,
                                      refinement_window_start,
                                      ):
        """Searches for the best candidate to be the last epoch in the re-
        finement window. In this context, the best candidate is the first
        epoch starting from the initial one to the next ones that has a
        high probability of being in the sleep state.

        Parameters
        ----------
        initial_candidate : int
                Location of the initial candidate to be the transition betwe-
                en from the awake to the sleep state.
        refinement_window_start : int
                Index of the first epoch of a refinement window.

        Returns
        -------
        refinement_window_end : int
                Index of the last epoch of a refinement window.
        """

        refinement_window_end = initial_candidate

        # The refinement window end will be defined based on 3 criteria. 
        # First: valid datetime gaps.
        valid_gap = self.datetime_gap_check(refinement_window_end)

        # Second: if gaps remain valid, the search continues until an e-
        # poch with a high enough proportion of zeros around it is found.
        zero_proportion_around_end = self.compute_zero_proportion_around_end(refinement_window_end)

        # Third: also, the refinement window ending in the chosen epoch must
        # have a sustained low level of median activity in the last epochs.
        refinement_window_activity_median = self.compute_refinement_window_median(refinement_window_start,refinement_window_end)

        metric = self.compute_metric(refinement_window_activity_median)        

        median_ending = refinement_window_activity_median[len(refinement_window_activity_median)-self.activity_median_analysis_window:len(refinement_window_activity_median)]
        median_ending_above_metric_proportion = 1-below_prop(median_ending,metric)
        while (   # Boundary conditions for the end of the bedtime refinement interval
                    (refinement_window_end+1 < self.data_length) and
                    (refinement_window_end+1 < self.next_transition) and
                    valid_gap and
                    # (metric < 0.5*self.minimum_short_window_activity_median_threshold) and
                    (
                        (zero_proportion_around_end < self.zero_proportion_threshold)  or
                        (median_ending_above_metric_proportion > 0)
                    )
              ):
            refinement_window_end += 1   # Interval is stretched to the right

            zero_proportion_around_end = self.compute_zero_proportion_around_end(refinement_window_end)

            valid_gap = self.datetime_gap_check(refinement_window_end)

            refinement_window_activity_median = self.compute_refinement_window_median(refinement_window_start,refinement_window_end)

            metric = self.compute_metric(refinement_window_activity_median)     

            median_ending = refinement_window_activity_median[len(refinement_window_activity_median)-self.activity_median_analysis_window:len(refinement_window_activity_median)]
            median_ending_above_metric_proportion = 1-below_prop(median_ending,metric)

        return refinement_window_end

    def bridge_gap_validation(self,
                              refinement_window_end,
                              ):
        """When the search for the best candidate to refinement window
        end is terminated by the discovery of an invalid datetime gap,
        this gap is "bridged over". In practice, the epoch after the gap
        will be defined as the new refinement window start and the se-
        arch for a new refinement window end is proceeded.

        Parameters
        ----------
        refinement_window_end : int
                Index of the last epoch of a refinement window.

        Returns
        -------
        refinement_window_start : int
                Index of the first epoch of a refinement window.
        refinement_window_end : int
                Index of the last epoch of a refinement window.
        """

        refinement_window_start = refinement_window_end+1
        if self.verbose:
            print("bridging the gap")
            print("start1",self.datetime_stamps[refinement_window_start])

        initial_candidate = refinement_window_start+1
        refinement_window_end = self.compute_refinement_window_end(initial_candidate,refinement_window_start)
        if self.verbose:
            print("end1",self.datetime_stamps[refinement_window_end])

        return refinement_window_start,refinement_window_end

    def compute_improved_refinement_window_start(self,
                                                 refinement_window_start,
                                                 refinement_window_end,
                                                 ):
        """After an initial refinement window has been successfully
        defined, it becomes possible to compute features related to
        the entire refinement window and those are utilized to impro-
        ve the refinment window start using new criteria.

        Parameters
        ----------
        refinement_window_start : int
                Index of the first epoch of a refinement window.
        refinement_window_end : int
                Index of the last epoch of a refinement window.

        Returns
        -------
        efinement_window_start : int
                Index of the first epoch of a refinement window.
        """


        # The improved refinement window start will be defined based on 2
        # criteria. First: valid datetime gaps, if an invalid gap is found
        # the epoch right after it is chosen, because the existence of a
        # gap is considered impossible when the subject is asleep.
        valid_gap = self.datetime_gap_check(refinement_window_start)


        # Third: also, the refinement window beginning in the chosen epoch 
        # must have a sustained high level of median activity in the first
        # epochs.
        refinement_window_activity_median = self.compute_refinement_window_median(refinement_window_start,refinement_window_end)

        metric = self.compute_metric(refinement_window_activity_median)     

        median_ending = refinement_window_activity_median[0:self.activity_median_analysis_window]
        median_ending_below_metric_proportion = below_prop(median_ending,metric)
        while (   
                   (refinement_window_start > 0) and
                   (refinement_window_start-1 > self.previous_transition) and 
                   valid_gap and
                   # (metric < 0.5*0.5*self.minimum_short_window_activity_median_threshold) and
                   (median_ending_below_metric_proportion > 0)
              ):
            refinement_window_start -= 1 

            if refinement_window_start > 0:
                valid_gap = self.datetime_gap_check(refinement_window_start)
                
            refinement_window_activity_median = self.compute_refinement_window_median(refinement_window_start,refinement_window_end)

            metric = self.compute_metric(refinement_window_activity_median)

            median_ending = refinement_window_activity_median[0:self.activity_median_analysis_window]
            median_ending_below_metric_proportion = below_prop(median_ending,metric)

        return refinement_window_start


    def remove_after_long_valley(self,
                                 peaks_and_valleys,
                                 ):
        """In the context of a bedtime refinement, when a long e-
        nough valley (low activity region) is found, it'll be un-
        derstood that it represents that the subject fell asleep.
        This leads to the conclusion that the bedtime was neces-
        sarily before this long valley and any peaks or valleys
        after it may be removed.

        Parameters
        ----------
        peaks_and_valleys: pd.DataFrame
                Contains the information of the location and fea-
                tures of peaks and valleys inside a refinement
                window.

        Returns
        -------
        peaks_and_valleys: pd.DataFrame
                Contains the information of the location and fea-
                tures of peaks and valleys inside a refinement
                window.
        """

        # Valleys that are too long are classified as invalid.
        peaks_and_valleys["valid"] = True
        peaks_and_valleys.loc[peaks_and_valleys["class"] == "v","valid"] = peaks_and_valleys.loc[peaks_and_valleys["class"] == "v","length"] < self.bedtime_high_probability_sleep_valley_length

        if self.verbose:
            print("metric",self.metric)
            print("start",self.datetime_stamps[self.refinement_window_start])
            print("end",self.datetime_stamps[self.refinement_window_end])
            print("peaks_and_valleys\n",peaks_and_valleys)

            wt = self.datetime_stamps[self.refinement_window_start:self.refinement_window_end+1]
            fs=16
            plt.figure()
            plt.plot(wt,self.refinement_window_levels[self.refinement_window_start:self.refinement_window_end+1],lw=3,color="black",label="levels")
            plt.plot(wt,self.refinement_window_activity,lw=0.5,label="self.activity")
            plt.plot(wt,self.refinement_window_activity_median,label="median")
            plt.plot(wt,np.insert(self.refinement_window_activity_median_difference_smoothed,0,0),label="diff")
            plt.plot(wt,self.metric*np.ones(len(self.refinement_window_activity)),linestyle="--",label="metric")
            plt.legend(fontsize=fs)
            plt.title("Unfiltered levels")
            # plt.show()

        if self.do_remove_after_long_valley:
            invalid_valleys = peaks_and_valleys[peaks_and_valleys["valid"] == False].index.to_numpy()
            invalid_valley_count = len(invalid_valleys)
            if invalid_valley_count > 0:
                peaks_and_valleys = peaks_and_valleys.loc[0:invalid_valleys[0],:]

            if self.verbose:
                print("remove long valley\n",peaks_and_valleys)

        return peaks_and_valleys

    def remove_before_long_peak(self,
                                 peaks_and_valleys,
                                 ):
        """In the context of a bedtime refinement, when a long e-
        nough peak (high activity region) is found, it'll be un-
        derstood that it represents that the subject was awake.
        This leads to the conclusion that the bedtime is necessa-
        rily after this long peak and any peaks or valleys befo-
        re it may be removed.

        Parameters
        ----------
        peaks_and_valleys: pd.DataFrame
                Contains the information of the location and fea-
                tures of peaks and valleys inside a refinement
                window.

        Returns
        -------
        peaks_and_valleys: pd.DataFrame
                Contains the information of the location and fea-
                tures of peaks and valleys inside a refinement
                window.
        """

        peaks_and_valleys_count = len(peaks_and_valleys)

        # Peaks that are too long are classified as invalid.
        peaks_and_valleys["valid"] = True
        peaks_and_valleys.loc[peaks_and_valleys["class"] == "p","valid"] = peaks_and_valleys.loc[peaks_and_valleys["class"] == "p","length"] < self.bedtime_high_probability_awake_peak_length
        if self.verbose:
            print("metric",self.metric)
            print("start",self.datetime_stamps[self.refinement_window_start])
            print("end",self.datetime_stamps[self.refinement_window_end])
            print("peaks_and_valleys\n",peaks_and_valleys)

        if self.do_remove_before_long_peak:
            invalid_peaks = peaks_and_valleys[peaks_and_valleys["valid"] == False].index.to_numpy()
            invalid_peak_count = len(invalid_peaks)
            if invalid_peak_count > 0:
                if invalid_peaks[invalid_peak_count-1] < peaks_and_valleys_count-1:
                    peaks_and_valleys = peaks_and_valleys.loc[invalid_peaks[invalid_peak_count-1]:,:]

            if self.verbose:
                print("remove long peak\n",peaks_and_valleys)

        return peaks_and_valleys

    def remove_before_tall_peak(self,
                       peaks_and_valleys,
                       ):
        """In the context of a bedtime refinement, a tall peak 
        is a region with a specially high median activity. When a
        region like this is found, it'll be understood that it re-
        presents that the subject was awake. This leads to the con-
        clusion that the bedtime is necessarily after this long 
        peak and any peaks or valleys before it may be removed.

        Parameters
        ----------
        peaks_and_valleys: pd.DataFrame
                Contains the information of the location and fea-
                tures of peaks and valleys inside a refinement
                window.

        Returns
        -------
        peaks_and_valleys: pd.DataFrame
                Contains the information of the location and fea-
                tures of peaks and valleys inside a refinement
                window.
        """

        # Peaks that are too tall are classified as invalid.
        peaks_and_valleys["valid"] = True
        if self.metric > 5:
            peaks_and_valleys.loc[peaks_and_valleys["class"]=="p","valid"] = peaks_and_valleys.loc[peaks_and_valleys["class"]=="p","mean"] <= 10*self.metric

        if self.do_remove_before_tall_peak:
            invalid_peaks = peaks_and_valleys[peaks_and_valleys["valid"] == False].index.to_numpy()
            invalid_peak_count = len(invalid_peaks)
            if invalid_peak_count > 0:
                peaks_and_valleys = peaks_and_valleys.loc[invalid_peaks[invalid_peak_count-1]:,:]

            if self.verbose:
                print("remove too tall peak\n",peaks_and_valleys)

        return peaks_and_valleys


    def filter_peaks_and_valleys(self,
                                 peaks_and_valleys,
                                 ):
        """Sequentially analyzes various combinations of features
        of the peaks and valleys present in the refinement window
        and removes the ones considered invalid.

        Parameters
        ----------
        peaks_and_valleys: pd.DataFrame
                Contains the information of the location and fea-
                tures of peaks and valleys inside a refinement
                window.

        Returns
        -------
        peaks_and_valleys: pd.DataFrame
                Contains the information of the location and fea-
                tures of peaks and valleys inside a refinement
                window.
        """

        peaks_and_valleys_count = len(peaks_and_valleys)
        peaks_and_valleys.index = range(peaks_and_valleys_count)
                
        if ((peaks_and_valleys_count > 2) and (peaks_and_valleys_count != 3)):
            # First region is treated separately and may be removed, 
            # if it's a valley.
            if (peaks_and_valleys.at[0,"class"] == "v"):
                if (peaks_and_valleys.at[0,"length"] < self.bedtime_minimum_valley_length):
                    # In this case, the first valley is merged into
                    # it's succeeding peak
                    peaks_and_valleys = remove_peak_valley(peaks_and_valleys,0,self.refinement_window_activity,self.metric)
                    peaks_and_valleys_count = len(peaks_and_valleys)

            region_index = 1
            while (region_index < peaks_and_valleys_count) and ((peaks_and_valleys_count > 2) and (peaks_and_valleys_count != 3)):
                remove = False
                # print("bt region_index",region_index)

                # All regions are subjected to length-based threshol-
                # ding and, if not filtered, they are evaluated ba-
                # sed off other criteria
                if (peaks_and_valleys.at[region_index,"class"] == "p"):
                    if region_index < peaks_and_valleys_count-2:
                        if (peaks_and_valleys.at[region_index,"length"] < self.bedtime_minimum_peak_length):
                            if (peaks_and_valleys.at[region_index+1,"length"] < self.bedtime_minimum_valley_length
                               ):
                               # Bedtime peaks with invalid lengths 
                               # are spared if they're right before 
                               # a valley with invalid length
                                pass
                            else:
                                remove = True

                        else:
                            # Bedtime peaks with valid lengths are 
                            # removed if their mean is small
                            if (peaks_and_valleys.at[region_index,"mean"] <= 1.33*self.metric):   
                                remove = True
                            
                            elif (
                                  (self.condition == 2) and 
                                  (
                                   (region_index > 0) and 
                                # Additional criteria: relative length
                                # of preceding valley
                                   (peaks_and_valleys.at[region_index-1,"length"] >= 15*peaks_and_valleys.at[region_index,"length"])
                                  )
                                 ):
                                remove = True

                    else:
                        # If a peak comes before the last region, 
                        # specific criteria are applied. If it's
                        # long and with a low level of activity,
                        # it will be removed.
                        if (
                            (peaks_and_valleys.at[region_index-1,"length"] >= 30)
                            and (
                                 (peaks_and_valleys.at[region_index-1,"zero_proportion"] > 2.0/3)
                                 or (peaks_and_valleys.at[region_index-1,"above_threshold_proportion"] < 0.1)
                                 )
                            ):
                            remove = True

                else:#(peaks_and_valleys.at[region_index,"class"] == "v")
                    if (region_index < peaks_and_valleys_count-1):
                        if (peaks_and_valleys.at[region_index,"length"] < self.bedtime_minimum_valley_length):
                            if region_index > 1:
                                remove = True

                            else:
                            # If a valley with an invalid length
                            # is the second region, it will be re-
                            # moved only if the next peak has a
                            # valid length and a low proportion
                            # of zeros.
                                if (
                                    (peaks_and_valleys.at[region_index+1,"length"] > self.bedtime_minimum_peak_length)
                                    and (peaks_and_valleys.at[region_index+1,"zero_proportion"] < 1.0/3)
                                    ):
                                    remove = True


                        else:
                            # Valleys with valid lengths may be 
                            # removed if their features are not
                            # good. This is evaluated by several
                            # criteria combinations.
                            remove_points = 0
                            if (peaks_and_valleys.at[region_index,"above_threshold_proportion"] >= 0.33):
                                remove_points += 1
                            if (peaks_and_valleys.at[region_index,"zero_proportion"] < 0.45):
                                if (peaks_and_valleys.at[region_index,"zero_proportion"] > 0.1):
                                    remove_points += 1
                                else:
                                    remove_points += 1.5

                            if (peaks_and_valleys.at[region_index,"mean"] >= 0.66*self.metric):
                                remove_points += 0.5
                            if (
                                (peaks_and_valleys.at[region_index,"length"]/len(self.refinement_window_activity) >= 0.3) or
                                (
                                  (region_index > 0) and 
                                  (peaks_and_valleys.at[region_index,"length"] >= 1.5*peaks_and_valleys.at[region_index-1,"length"])
                                )
                               ):
                                remove_points -= 1

                            if remove_points > 1.5:
                                remove = True

                if remove:
                    # When a valley or a peak is removed, it's
                    # actually merged into the neighboring re-
                    # gions to create a large new region com-
                    # posed by the 3 regions. 
                    peaks_and_valleys = remove_peak_valley(peaks_and_valleys,region_index,self.refinement_window_activity,self.metric)
                    peaks_and_valleys_count = len(peaks_and_valleys)

                    if self.verbose:    
                        print("intermediate peaks_and_valleys region_index",region_index,"\n",peaks_and_valleys)

                else:
                    region_index += 1

        return peaks_and_valleys


    def identify_bedtime_candidates(self,
                                    peaks_and_valleys,
                                    ):
        """In the context of a bedtime refinement, bedtime candi-
        dates will be the first epochs of the valley regions, pre-
        ferably if those regions have significantly lower activi-
        ty levels than their preceding peaks.

        Parameters
        ----------
        peaks_and_valleys: pd.DataFrame
                Contains the information of the location and fea-
                tures of peaks and valleys inside a refinement
                window.

        Returns
        -------
        bedtime_candidates: np.array [int]
                Indexes of the bedtime candidate epochs.
        """

        bedtime_candidates = []
        peaks_and_valleys_count = len(peaks_and_valleys)
        for region_index in range(peaks_and_valleys_count):
            start = int(self.refinement_window_start + peaks_and_valleys.at[region_index,"start"])
            end = int(self.refinement_window_start + peaks_and_valleys.at[region_index,"end"])
            self.refinement_window_levels[start:end] = peaks_and_valleys.at[region_index,"mean"]

            if (peaks_and_valleys.at[region_index,"class"] == "v"):
                if (region_index > 1) and (region_index < peaks_and_valleys_count-1):
                    if (peaks_and_valleys.at[region_index,"mean"] < 0.5*peaks_and_valleys.at[region_index-1,"mean"]):
                        bedtime_candidates.append(int(peaks_and_valleys.at[region_index,"start"]))
                else:
                    bedtime_candidates.append(int(peaks_and_valleys.at[region_index,"start"]))

        # If there weren't any valleys that fulfill the previous
        # criteria, all will be used.
        bedtime_candidates_count = len(bedtime_candidates)
        if (bedtime_candidates_count == 0):
            bedtime_candidates = peaks_and_valleys[peaks_and_valleys["class"] == "v"].loc[:,"start"].to_numpy().astype(int)

        bedtime_candidates = np.array(bedtime_candidates)

        return bedtime_candidates


    def bedtime_candidates_crossings_filter(self,
                                            bedtime_candidates,
                                            ):
        """In the context of a bedtime refinement window, when
        the short-window-median-filtered activity stabilizes on
        a low level it'll be understood that it represents that
        the subject fell asleep. This leads to the conclusion 
        that the bedtime was necessarily before this stabiliza-
        tion and any peaks or valleys after it may be removed.

        Parameters
        ----------
        bedtime_candidates: np.array [int]
                Indexes of the bedtime candidate epochs.

        Returns
        -------
        bedtime_candidates: np.array [int]
                Indexes of the bedtime candidate epochs.
        short_window_filter_metric_crossing_down: np.array [int]
                Indexes of the epochs where the short-window-me-
                dian-filtered activity goes lower than the me-
                tric.
        """

        # First step: identify all the epochs where the short-
        # window-median-filtered activity crosses the metric
        # threshold.
        short_window_filter_metric_crossing = np.where(self.short_window_activity_median[self.refinement_window_start:self.refinement_window_end+1] >= self.metric, 1, 0)
        short_window_filter_metric_crossing = np.diff(np.concatenate(([0],short_window_filter_metric_crossing)))

        # Second step: separate crossing-up's from crossing-
        # down's
        short_window_filter_metric_crossing_up = (short_window_filter_metric_crossing > 0).nonzero()[0]
        short_window_filter_metric_crossing_down = (short_window_filter_metric_crossing < 0).nonzero()[0]

        if self.verbose:
            print("short_window_filter_metric_crossing_up",short_window_filter_metric_crossing_up)
            print("short_window_filter_metric_crossing_down",short_window_filter_metric_crossing_down)

        if len(short_window_filter_metric_crossing_up) > 0:
            # If there are multiple metric crossing-up's, all
            # candidates preceding the last crossing are eli-
            # minated
            last_short_window_filter_metric_crossing_up = short_window_filter_metric_crossing_up[len(short_window_filter_metric_crossing_up)-1]
            mask = np.where(bedtime_candidates >= last_short_window_filter_metric_crossing_up,True,False)
            bedtime_candidates = bedtime_candidates[mask]
            if self.verbose:
                print("bedtime_candidates_crossings_filter bedtime_candidates\n",bedtime_candidates)
                

        return bedtime_candidates, short_window_filter_metric_crossing_down

    
    def choose_best_bedtime_candidate(self,
                                      bedtime_candidates,
                                      short_window_filter_metric_crossing_down,
                                      ):
        """Assigns scores to the bedtime candidates based on their
        features and location in order to determine which is one
        is most likely the actual transition between awake and 
        asleep states.

        Parameters
        ----------
        bedtime_candidates: np.array [int]
                Indexes of the bedtime candidate epochs.
        short_window_filter_metric_crossing_down: float
                Indexes of the epochs where the short-window-me-
                dian-filtered activity goes lower than the me-
                tric.

        Returns
        -------
        refined_bedtime : int
                Index of the refined bedtime epoch
        """

        bedtime_candidates_count = len(bedtime_candidates)
        if bedtime_candidates_count > 0:
            bedtime_candidate_scores = pd.DataFrame(bedtime_candidates,columns=["candidate"])
            bedtime_candidate_scores["datetime_stamp"] = [self.datetime_stamps[self.refinement_window_start+bedtime_candidates[candidate]] for candidate in range(bedtime_candidates_count)]

            # The presence of a activity median difference valley
            # near a candidate is indicative of a sudden drop in 
            # the level of activity, this is associated with the 
            # transition from awake to asleep.
            bedtime_candidate_scores["activity_median_difference"] = [get_peak(self.refinement_window_activity_median_difference_smoothed,bedtime_candidates[candidate],True) for candidate in range(bedtime_candidates_count)]
            
            # In the context of bedtime detection, the best can-
            # didates have low numbers of epochs with activity a-
            # bove metric.
            bedtime_candidate_scores["epochs_above_metric_after"] = self.after_candidate_window

            # If the number of epochs with activity above metric
            # is below the user-set threshold, this will be True.
            bedtime_candidate_scores["thresholded"] = False

            # This indicates if there's any invalid gaps near the
            # candidate. Before evaluation, all are initialized 
            # as invalid.
            bedtime_candidate_scores["gap_after"] = True

            # As features are evaluated, the candidates will be
            # scored.
            bedtime_candidate_scores["score"] = 0.0

            # Being the last (relative to datetime) is an interes-
            # ting feature for the candidate, so it may be scored.
            if self.bedtime_score_last_candidate:
                bedtime_candidate_scores.at[bedtime_candidates_count-1,"score"] += self.bedtime_last_candidate_score
                if self.verbose:
                    print("self.bedtime_score_last_candidate bedtime_candidate_scores\n",bedtime_candidate_scores)

            # The candidate that's closest to the last short-win-
            # down-median-filtered activity metric crossing-down 
            # gets an extra score.
            if len(short_window_filter_metric_crossing_down) > 0:
                short_window_filter_metric_crossing_down = short_window_filter_metric_crossing_down[len(short_window_filter_metric_crossing_down)-1]
                metric_crossing_distance = np.absolute(bedtime_candidates - short_window_filter_metric_crossing_down)
                best_crossing_distance_candidate = np.argmin(metric_crossing_distance)
                bedtime_candidate_scores.at[best_crossing_distance_candidate,"score"] += self.bedtime_best_crossing_distance_candidate_score
                if self.verbose:
                    print("short_window_filter_metric_crossing_down",short_window_filter_metric_crossing_down)
                    print("best_crossing_distance_candidate",best_crossing_distance_candidate)
                    print("bedtime_candidate_scores\n",bedtime_candidate_scores)

            # The candidate with the most abrupt valley in median
            # activity difference gets an extra score.
            bedtime_candidate_scores.sort_values(by=["activity_median_difference",],axis=0,ignore_index=True,inplace=True)
            bedtime_candidate_scores.at[0,"score"] += self.bedtime_best_median_difference_candidate_score

            if self.verbose:
                print("bedtime_candidate_scores\n",bedtime_candidate_scores)

            # Lastly, the neighborhood of each candidate is evalu-
            # ated and scores are given based on their number of
            # following epochs above metric
            if self.condition in [0,2]:
                candidate_index = 0
                while (candidate_index < bedtime_candidates_count):
                    candidate = self.refinement_window_start + bedtime_candidate_scores.at[candidate_index,"candidate"]

                    after_candidate_window_end = candidate+self.after_candidate_window
                    if after_candidate_window_end > self.data_length:
                        after_candidate_window_end = self.data_length
                    activity_after_candidate = self.activity[candidate:after_candidate_window_end]

                    valid_gap = True
                    g = candidate
                    while valid_gap and (g+1 < after_candidate_window_end):
                        valid_gap,datetime_seconds_gap = self.datetime_gap_check(g,direction="forward",return_gap=True)
                        if not valid_gap:
                            if self.verbose:
                                print("gap",datetime_seconds_gap)
                        g += 1
                    
                    if valid_gap:
                        bedtime_candidate_scores.at[candidate_index,"gap_after"] = False

                        epochs_above_metric_after = np.where(activity_after_candidate >= self.metric,1,0)   # And evaluate how many epochs inside the window are above the metric
                        epochs_above_metric_after = np.sum(epochs_above_metric_after)

                        bedtime_candidate_scores.at[candidate_index,"epochs_above_metric_after"] = epochs_above_metric_after
                        if epochs_above_metric_after <= self.bedtime_maximum_epochs_above_metric_after_candidate:
                            bedtime_candidate_scores.at[candidate_index,"thresholded"] = True

                    candidate_index += 1

                thresholded_bedtime_candidates = bedtime_candidate_scores[bedtime_candidate_scores["thresholded"]]
                if self.verbose:
                    print("bedtime_candidate_scores\n",bedtime_candidate_scores)
                    print("thresholded_bedtime_candidates\n",thresholded_bedtime_candidates)
                    

                if len(bedtime_candidate_scores[bedtime_candidate_scores["gap_after"]==False]) > 0:
                    if len(thresholded_bedtime_candidates) == 0:   # If there aren't any valid candidates, the one with the least above-metric epochs is chosen
                        bedtime_candidate_scores.sort_values(by=["gap_after","epochs_above_metric_after",],axis=0,ignore_index=True,inplace=True)
                        bedtime_candidate_scores.at[0,"score"] += self.bedtime_best_epochs_above_metric_after_score

                    else:
                        if self.bedtime_maximum_epochs_above_metric_after_candidate > 0:
                            thresholded_score_factor = (-self.bedtime_thresholded_candidate_score_amplitude/self.bedtime_maximum_epochs_above_metric_after_candidate)
                        else:
                            thresholded_score_factor = (-self.bedtime_thresholded_candidate_score_amplitude/1e-3)

                        thresholded_bedtime_candidates.loc[:,"score"] = thresholded_score_factor*(thresholded_bedtime_candidates.loc[:,"epochs_above_metric_after"]-self.bedtime_maximum_epochs_above_metric_after_candidate)

                        if self.verbose:
                            print("scored thresholded_candidates\n",thresholded_bedtime_candidates["score"])

                        bedtime_candidate_scores.loc[bedtime_candidate_scores["thresholded"],"score"] += thresholded_bedtime_candidates["score"]+self.bedtime_thresholded_candidate_score_minimum

                if self.verbose:
                    print("bedtime_candidate_scores\n",bedtime_candidate_scores)

                bedtime_candidate_scores.sort_values(by=["score","candidate"],axis=0,ascending=False,ignore_index=True,inplace=True)
                if self.verbose:
                    print("bedtime_candidate_scores\n",bedtime_candidate_scores)
                    print("top_grader",bedtime_candidate_scores.at[0,"candidate"])

                # The highest scored candidate is chosen as the refi-
                # ned bedtime.
                if ((self.consider_second_best_candidate) and (bedtime_candidates_count > 1)):
                        if (
                                (bedtime_candidate_scores.at[0,"candidate"]-bedtime_candidate_scores.at[1,"candidate"] > 60) and 
                                (bedtime_candidate_scores.at[1,"thresholded"] and (not bedtime_candidate_scores.at[0,"thresholded"]))
                            ):
                                if self.verbose:
                                        print("second best")
                                refined_bedtime = self.refinement_window_start + int(bedtime_candidate_scores.at[1,"candidate"])  
                        else:
                                refined_bedtime = self.refinement_window_start + int(bedtime_candidate_scores.at[0,"candidate"])  
                else:
                        refined_bedtime = self.refinement_window_start + int(bedtime_candidate_scores.at[0,"candidate"])                    

            else:
                refined_bedtime = self.refinement_window_start + bedtime_candidates[-1]

        else:
            refined_bedtime = self.initial_transition_candidate    # Mantains if no edges are detected

        if self.verbose:
            print("refined_bedtime",self.datetime_stamps[refined_bedtime])
            # input()

        return refined_bedtime


    def refine(self,
               refinement_window_levels,
               initial_transition_candidate,
               previous_transition,
               next_transition,
               verbose=False,
               ):
        """Searches for the epoch is most likely the actual transition betwe-
        en awake and asleep states. To do so, first: a refinement window (a
        region that starts in the awake state and ends in the asleep state)
        will be determined and then: the best possible transition is searched
        for inside it.

        Parameters
        ----------
        refinement_window_levels : np.array [float]
                Mean activity levels inside the refinement window.
        initial_transition_candidate : int
                Index of the initial bedtime candidate.
        previous_transition : int
                Index of the preceding getup time.
        next_transition : int
                Index of the succeeding getup time.
        verbose: int or boolean, default False
                Verbosity level

        Returns
        -------
        refined_bedtime : int
                Index of the refined bedtime epoch
        refinement_window_start : int
                Index of the first epoch of a refinement window.
        refinement_window_end : int
                Index of the last epoch of a refinement window.
        refinement_window_activity_median : np.array [float]
                Median-filtered activity in the refinement window.
        refinement_window_activity_median_difference_smoothed : np.array 
        [float]
                Smoothing-filtered median activity differences in the refi-
                nement window.
        refinement_window_levels : np.array [float]
                Mean activity levels inside the refinement window.
        metric : float
                Threshold of separation between high and low activity in the
                context of a refinement window.
        """

        self.refinement_window_levels = refinement_window_levels
        self.initial_transition_candidate = initial_transition_candidate
        self.previous_transition = previous_transition
        self.next_transition = next_transition
        self.verbose = verbose

        if self.verbose:
            print("\n\n\nbed time refinement")
            print("initial",self.datetime_stamps[self.initial_transition_candidate])

        initial_candidate = self.initial_transition_candidate-1
        refinement_window_start = self.compute_initial_refinement_window_start(initial_candidate)        
        if self.verbose:
            print("start0",self.datetime_stamps[refinement_window_start])

        initial_candidate = self.initial_transition_candidate+1
        refinement_window_end = self.compute_refinement_window_end(initial_candidate,refinement_window_start)
        if self.verbose:
            print("end0",self.datetime_stamps[refinement_window_end])

        valid_gap = self.datetime_gap_check(refinement_window_end)
        if not valid_gap:
            refinement_window_start,refinement_window_end = self.bridge_gap_validation(refinement_window_end)  

        else:
            refinement_window_start = self.compute_improved_refinement_window_start(refinement_window_start,refinement_window_end)
            if self.verbose:
                print("start1",self.datetime_stamps[refinement_window_start])     


        refinement_window_datetime_difference = datetime_diff(self.datetime_stamps[refinement_window_start:refinement_window_end+1])
        refinement_window_datetime_gaps = np.where(refinement_window_datetime_difference >= self.maximum_allowed_gap,1,0)
        if np.sum(refinement_window_datetime_gaps) > 0:
            first_gap_location = list(refinement_window_datetime_gaps).index(1)
            if first_gap_location > len(refinement_window_datetime_difference)-first_gap_location:
                refinement_window_end = refinement_window_start + first_gap_location - 1
                # print("end1",self.datetime_stamps[refinement_window_end])
            else:
                refinement_window_start = refinement_window_start + first_gap_location + 1
                # print("start2",self.datetime_stamps[refinement_window_start])


        if refinement_window_start >= refinement_window_end:
            refinement_window_end = refinement_window_start + 60
            if refinement_window_end >= self.data_length:
                refinement_window_end = self.data_length-1

        self.refinement_window_start = refinement_window_start
        self.refinement_window_end = refinement_window_end
        self.refinement_window_activity_median = self.compute_refinement_window_median(self.refinement_window_start,self.refinement_window_end)

        self.metric = self.compute_metric(self.refinement_window_activity_median)     

        self.refinement_window_activity_median_difference = np.diff(self.refinement_window_activity_median)
        self.refinement_window_activity_median_difference_smoothed = median_filter(self.refinement_window_activity_median_difference,self.median_filter_half_window_size)

        self.refinement_window_activity = self.activity[self.refinement_window_start:self.refinement_window_end+1]    # Raw activity in the refinement interval
        peaks_and_valleys = identify_peaks_and_valleys(self.refinement_window_activity,self.refinement_window_activity_median,self.metric)   # Transitions relative to the median filter and chosen metric
        peaks_and_valleys_count = len(peaks_and_valleys)

        # print("initial peaks_and_valleys")
        # print(peaks_and_valleys)

        for region_index in range(peaks_and_valleys_count):
            start = int(self.refinement_window_start + peaks_and_valleys.at[region_index,"start"])
            end = int(self.refinement_window_start + peaks_and_valleys.at[region_index,"end"])
            self.refinement_window_levels[start:end] = peaks_and_valleys.at[region_index,"mean"]

        peaks_and_valleys = self.remove_after_long_valley(peaks_and_valleys)
        peaks_and_valleys_count = len(peaks_and_valleys)
        peaks_and_valleys.index = range(peaks_and_valleys_count)

        peaks_and_valleys = self.remove_before_long_peak(peaks_and_valleys)
        peaks_and_valleys_count = len(peaks_and_valleys)
        peaks_and_valleys.index = range(peaks_and_valleys_count)

        peaks_and_valleys = self.remove_before_tall_peak(peaks_and_valleys)
        peaks_and_valleys_count = len(peaks_and_valleys)
        peaks_and_valleys.index = range(peaks_and_valleys_count)
        
        self.peaks_and_valleys = self.filter_peaks_and_valleys(peaks_and_valleys)
        if self.verbose:
            print("filtered peaks_and_valleys\n",self.peaks_and_valleys)

        peaks_and_valleys_count = len(self.peaks_and_valleys)

        if self.verbose:
                print("initial borders")
                print(self.refinement_window_start)
                print(self.refinement_window_end)
        
        refinement_window_start = self.refinement_window_start+self.peaks_and_valleys.at[0,"start"]
        self.refinement_window_end = self.refinement_window_start+self.peaks_and_valleys.at[peaks_and_valleys_count-1,"end"]
        if self.refinement_window_end >= self.data_length:
            self.refinement_window_end = self.data_length-1
        self.refinement_window_start = refinement_window_start

        if self.verbose:
                print("filtered borders")
                print(self.refinement_window_start)
                print(self.refinement_window_end)
                print("")

        self.refinement_window_activity_median = self.compute_refinement_window_median(self.refinement_window_start,self.refinement_window_end)

        self.metric = self.compute_metric(self.refinement_window_activity_median)     

        self.refinement_window_activity_median_difference = np.diff(self.refinement_window_activity_median)
        self.refinement_window_activity_median_difference_smoothed = median_filter(self.refinement_window_activity_median_difference,self.median_filter_half_window_size)

        self.refinement_window_activity = self.activity[self.refinement_window_start:self.refinement_window_end+1]

        if self.update_peaks_and_valleys:
                self.peaks_and_valleys = identify_peaks_and_valleys(self.refinement_window_activity,self.refinement_window_activity_median,self.metric)
                if self.verbose:
                        print("updated peaks_and_valleys")
                        print(self.peaks_and_valleys)
        else:
                self.peaks_and_valleys.at[0,"start"] = 0
                self.peaks_and_valleys.at[0,"end"] = self.peaks_and_valleys.at[0,"start"]+self.peaks_and_valleys.at[0,"length"]
                for i in range(1,peaks_and_valleys_count):
                    self.peaks_and_valleys.at[i,"start"] = self.peaks_and_valleys.at[i-1,"end"]
                    self.peaks_and_valleys.at[i,"end"] = self.peaks_and_valleys.at[i,"start"]+self.peaks_and_valleys.at[i,"length"]


        self.bedtime_candidates = self.identify_bedtime_candidates(self.peaks_and_valleys)
        if self.verbose:
            print("bedtime_candidates\n",self.bedtime_candidates)

        filtered_bedtime_candidates, short_window_filter_metric_crossing_down = self.bedtime_candidates_crossings_filter(self.bedtime_candidates)
        if self.do_bedtime_candidates_crossings_filter:
                self.bedtime_candidates = filtered_bedtime_candidates

        self.refined_bedtime = self.choose_best_bedtime_candidate(self.bedtime_candidates,short_window_filter_metric_crossing_down)

        return self.refined_bedtime,self.refinement_window_start,self.refinement_window_end,self.refinement_window_activity_median,self.refinement_window_activity_median_difference_smoothed,self.refinement_window_levels,self.metric