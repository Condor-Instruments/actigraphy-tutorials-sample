# -*- coding: utf-8 -*-

# CSPD Getup time Refinement algorithm
# Julius Andretti

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


class CSPD_GetUpTime_Refiner:
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
                 getuptime_metric_method,
                 getuptime_metric_parameter,
                 condition,

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

                 do_remove_after_long_tall_peak,
                 getuptime_high_probability_awake_peak_length,
                 getuptime_high_probability_sleep_valley_length,
                 do_remove_before_long_valley,
                 update_peaks_and_valleys,
                 getuptime_score_first_candidate,
                 ):
        """Searches for the epoch that shows the best features to represent
        the transition from sleep (in-bed) to awake (out-of-bed) state, the
        "getup time".

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
                Minimum proportion of zeros for a valid refinement window 
                start candidate.
        getuptime_metric_method : {1, 2}
                If 1: the threshold will be computed as an user-set fraction 
                of the mean non-zero activity median in the refinement window.
                If 2: the threshold will be computed as an user-set quantile
                of the non-zero activity median in the refinement window.
        getuptime_metric_parameter : float
                If getuptime_metric_method==1: fraction of the mean non-zero activity 
                median in the refinement window to define as metric.
                If getuptime_metric_method==2: quantile of the non-zero activity median 
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

        getuptime_high_probability_sleep_valley_length : int
                If a valley with at least this length is identified, it'll
                be understood that it's highly probable that the subject is 
                asleep inside it.
        getuptime_high_probability_awake_peak_length : int
                If a peak with at least this length is identified, it'll
                be understood that it's highly probable that the subject is 
                awake inside it.
        do_remove_after_long_tall_peak : boolean
                If True, peaks and valleys after the last identified long
                tall peak will be removed.
        do_remove_before_long_valley : boolean
                If True, peaks and valleys before the last identified long
                valley will be removed.


        getuptime_minimum_valley_length : int
                Valleys shorter than this parameter may be removed.
        getuptime_minimum_peak_length : int
                Peaks shorter than this parameter may be removed.

        getuptime_score_first_candidate : boolean
                If True, the first candidate will get an extra score.
        getuptime_first_candidate_score : float
                Extra score for the first candidate.
        getuptime_best_median_difference_candidate_score : float
                Extra score for the candidate nearest to the sharpest rise
                in median activity.
        getuptime_best_crossing_distance_candidate_score : float
                Extra score for the candidate nearest to the last short-
                window-median-filtered activity metric crossing-up.

        after_candidate_window : int
                Length of the window of epochs after the candidate that
                will be analyzed to determine how many are above the me-
                tric threshold.
        getuptime_maximum_epochs_above_metric_after_candidate : int
                Candidates with fewer epoch above metric after them than
                this parameter ("thresholded") will get extra scores.
        getuptime_best_epochs_above_metric_after_score : float
                If there aren't any candidates with few enough epochs a-
                bove metric following, the one with less gets this extra
                score.
        getuptime_thresholded_candidate_score_amplitude : float
                Maximum (before adding the minimum score) extra score for 
                a "thresholded" candidate.
        getuptime_thresholded_candidate_score_minimum : float
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
        self.getuptime_metric_method = getuptime_metric_method
        self.getuptime_metric_parameter = getuptime_metric_parameter
        self.condition = condition

        self.getuptime_first_candidate_score = getuptime_first_candidate_score
        self.getuptime_best_median_difference_candidate_score = getuptime_best_median_difference_candidate_score
        self.getuptime_best_crossing_distance_candidate_score = getuptime_best_crossing_distance_candidate_score
        self.getuptime_best_epochs_above_metric_after_score = getuptime_best_epochs_above_metric_after_score
        self.getuptime_thresholded_candidate_score_amplitude = getuptime_thresholded_candidate_score_amplitude
        self.getuptime_thresholded_candidate_score_minimum = getuptime_thresholded_candidate_score_minimum
        self.getuptime_minimum_peak_length = getuptime_minimum_peak_length
        self.getuptime_minimum_valley_length = getuptime_minimum_valley_length

        self.after_candidate_window = after_candidate_window
        self.getuptime_maximum_epochs_above_metric_after_candidate = getuptime_maximum_epochs_above_metric_after_candidate
        self.zero_proportion_threshold = zero_proportion_threshold

        self.do_remove_after_long_tall_peak = do_remove_after_long_tall_peak
        self.getuptime_high_probability_awake_peak_length = getuptime_high_probability_awake_peak_length

        self.getuptime_high_probability_sleep_valley_length = getuptime_high_probability_sleep_valley_length
        self.do_remove_before_long_valley = do_remove_before_long_valley

        self.update_peaks_and_valleys = update_peaks_and_valleys

        self.getuptime_score_first_candidate = getuptime_score_first_candidate


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
        if refinement_window_start-2*self.median_filter_half_window_size >= 0:
            if refinement_window_end+2*self.median_filter_half_window_size+1 <= self.data_length:
                refinement_window_activity = self.activity[refinement_window_start-2*self.median_filter_half_window_size:refinement_window_end+2*self.median_filter_half_window_size+1]
            else:
                refinement_window_activity = self.activity[refinement_window_start-self.median_filter_half_window_size:self.data_length]
                while len(refinement_window_activity) < (refinement_window_end+1+4*self.median_filter_half_window_size-refinement_window_start):
                    refinement_window_activity = np.append(refinement_window_activity,np.max(self.activity[refinement_window_start:refinement_window_end+1]))
        else:
            refinement_window_activity = self.activity[0:refinement_window_end+2*self.median_filter_half_window_size+1]
            while len(refinement_window_activity) < (refinement_window_end+1+4*self.median_filter_half_window_size-refinement_window_start):
                refinement_window_activity = np.insert(refinement_window_activity, 0, np.max(self.activity[refinement_window_start:refinement_window_end+1]))

        refinement_window_activity_median = median_filter(refinement_window_activity,self.median_filter_half_window_size,padding='padded')

        return refinement_window_activity_median


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
            mean = np.mean(refinement_window_positive_activity_median)    # Mean of non-zero elements in filtered array            
            if self.getuptime_metric_method == 1:
                fraction_mean = self.getuptime_metric_parameter*mean   # User-set self.getuptime_metric_parameter of the mean metric
                metric = fraction_mean
            elif self.getuptime_metric_method == 2:
                p = np.quantile(refinement_window_positive_activity_median, self.getuptime_metric_parameter, interpolation='linear')   # Quantile metric
                metric = p

        return metric

    def compute_zero_proportion_around_start(self,
                                             refinement_window_start,
                                             ):
        """Computes the proportion of zeros in a window centered around the 
        first epoch of a refinement window.

        Parameters
        ----------
        refinement_window_start : int
                Index of the first epoch of a refinement window.

        Returns
        -------
        zero_proportion_around_end : float
                The computed proportion of zeros.
        """

        if ((refinement_window_start+self.half_window_around_border) < self.data_length):
            window_around_border_end = refinement_window_start+self.half_window_around_border+1
        else:
            window_around_border_end = self.data_length

        if ((refinement_window_start-self.half_window_around_border) > 0):
            window_around_border_start = refinement_window_start-self.half_window_around_border
        else:
            window_around_border_start = 0

        activity_around_start = self.activity[window_around_border_start:window_around_border_end]
        zero_proportion_around_start = zero_prop(activity_around_start)

        return zero_proportion_around_start


    def compute_refinement_window_end(self,
                                      initial_candidate,
                                      ):
        """Searches for the best candidate to be the last epoch in the re-
        finement window. In this context, the best candidate is the first
        epoch starting from the initial one to the next ones that has a
        high probability of being in the awake state.

        Parameters
        ----------
        initial_candidate : int
                Location of the initial candidate to be the transition betwe-
                en from the sleep to the awake state.

        Returns
        -------
        refinement_window_end : int
                Index of the last epoch of a refinement window.
        """

        refinement_window_end = initial_candidate

        # The initial refinement window end will be defined based on 2
        # criteria. First: valid datetime gaps, if an invalid gap is found
        # the epoch right before it is chosen, because the existence of a
        # gap is considered impossible when the subject is asleep.
        valid_gap = self.datetime_gap_check(refinement_window_end)

        # Second: if gaps remain valid, the search continues until an e-
        # poch with a high enough level of activity before it is found.
        short_window_activity_median_before = self.short_window_activity_median[refinement_window_end-self.short_window_activity_median_minimum_high_epochs:refinement_window_end] 
        high_short_window_activity_median_before_proportion = 1-below_prop(short_window_activity_median_before,self.quantile_threshold)

        while (   # Boundary conditions for the end of the getup time refinement interval
                   (refinement_window_end+1 < self.data_length) and 
                   (refinement_window_end+1 < self.next_transition) and
                   valid_gap and
                   (high_short_window_activity_median_before_proportion < 1)
              ):
            refinement_window_end += 1   # Interval is stretched to the right
            # print("gt refinement_window_end",refinement_window_end)

            valid_gap = self.datetime_gap_check(refinement_window_end)

            short_window_activity_median_before = self.short_window_activity_median[refinement_window_end-self.short_window_activity_median_minimum_high_epochs:refinement_window_end] 
            high_short_window_activity_median_before_proportion = 1-below_prop(short_window_activity_median_before,self.quantile_threshold)

        return refinement_window_end

    def compute_refinement_window_start(self,
                                        initial_candidate,
                                        refinement_window_end,
                                        ):
        """Searches for the best candidate to be the first epoch in the re-
        finement window. In this context, the best candidate is the first
        epoch starting from the initial one to the previous ones that has a
        high probability of being in the asleep state.

        Parameters
        ----------
        initial_candidate : int
                Location of the initial candidate to be the transition betwe-
                en from the sleep to the awake state.
        refinement_window_end : int
                Index of the last epoch of a refinement window.

        Returns
        -------
        refinement_window_start : int
                Index of the first epoch of a refinement window.
        """

        if self.verbose:
            print("compute_refinement_window_start")

        refinement_window_start = initial_candidate
        if refinement_window_start < 0:
            refinement_window_start = 0
    

        # The refinement window start will be defined based on 3 criteria. 
        # First: valid datetime gaps.
        valid_gap = self.datetime_gap_check(refinement_window_start)

        # Second: if gaps remain valid, the search continues until an e-
        # poch with a high enough proportion of zeros around it is found.
        zero_proportion_around_start = self.compute_zero_proportion_around_start(refinement_window_start)

        # Third: also, the refinement window ending in the chosen epoch must
        # have a sustained low level of median activity in the last epochs.
        refinement_window_activity_median = self.compute_refinement_window_median(refinement_window_start,refinement_window_end)

        metric = self.compute_metric(refinement_window_activity_median)

        median_ending = refinement_window_activity_median[0:self.activity_median_analysis_window]
        median_ending_above_metric_proportion = 1-below_prop(median_ending,metric)
        while (   # Boundary conditions for the end of the getup time refinement interval
                   (refinement_window_start > 0) and 
                   (refinement_window_start-1 > self.previous_transition) and 
                   valid_gap and
                   # (metric < 3) and
                   (
                       (zero_proportion_around_start < self.zero_proportion_threshold) or
                       (median_ending_above_metric_proportion > 0)
                   )
              ):
            refinement_window_start -= 1    # Interval is stretched to the left
            # print("gt refinement_window_start",refinement_window_start)

            if refinement_window_start > 0:
                valid_gap = self.datetime_gap_check(refinement_window_start)

            refinement_window_activity_median = self.compute_refinement_window_median(refinement_window_start,refinement_window_end)

            metric = self.compute_metric(refinement_window_activity_median)

            median_ending = refinement_window_activity_median[0:self.activity_median_analysis_window]
            median_ending_above_metric_proportion = 1-below_prop(median_ending,metric)

            zero_proportion_around_start = self.compute_zero_proportion_around_start(refinement_window_start)

        return refinement_window_start


    def bridge_gap_validation(self,
                              refinement_window_start,
                              ):
        """When the search for the best candidate to refinement window
        start is terminated by the discovery of an invalid datetime gap,
        this gap is "bridged over". In practice, the epoch before the gap
        will be defined as the new refinement window end and the se-
        arch for a new refinement window start is proceeded.

        Parameters
        ----------
        refinement_window_start : int
                Index of the first epoch of a refinement window.

        Returns
        -------
        refinement_window_start : int
                Index of the first epoch of a refinement window.
        refinement_window_end : int
                Index of the last epoch of a refinement window.
        """

        refinement_window_end = refinement_window_start-1
        if self.verbose:
            print("bridging the gap")
            print("end1",self.datetime_stamps[refinement_window_end])

        initial_candidate = refinement_window_start-2
        refinement_window_start = self.compute_refinement_window_start(initial_candidate,refinement_window_end)
        if self.verbose:
            print("start1",self.datetime_stamps[refinement_window_start])

        return refinement_window_start,refinement_window_end


    def remove_after_long_tall_peak(self,
                                    peaks_and_valleys,
                                    ):
        """In the context of a getuptime refinement, a tall peak 
        is a region with a specially high median activity. When a
        region like this that is also long is found, it'll be un-
        derstood that it represents that the subject was awake. 
        This leads to the conclusion that the getuptime is neces-
        sarily before this long tall peak and any peaks or valleys 
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

        peaks_and_valleys_count = len(peaks_and_valleys)

        # Long tall peaks are classified as invalid.
        peaks_and_valleys["valid"] = True
        if self.metric > 5:
            for region in range(peaks_and_valleys_count):
                if peaks_and_valleys.at[region,"class"] == "p":
                    if (peaks_and_valleys.at[region,"length"] >= self.getuptime_high_probability_awake_peak_length) and (peaks_and_valleys.at[region,"mean"] > 10*self.metric):
                        peaks_and_valleys.at[region,"valid"] = False

        if self.do_remove_after_long_tall_peak:
            invalid_regions = peaks_and_valleys[peaks_and_valleys["valid"] == False].index.to_numpy()
            invalid_region_count = len(invalid_regions)
            if invalid_region_count > 0:
                peaks_and_valleys = peaks_and_valleys.loc[:invalid_regions[0],:]

            if self.verbose:
                print("remove too tall peak\n",peaks_and_valleys)

        return peaks_and_valleys


    def remove_before_long_valley(self,
                                  peaks_and_valleys,
                                  ):
        """In the context of a getuptime refinement, when a long 
        enough valley (low activity region) is found, it'll be un-
        derstood that it represents that the subject was asleep.
        This leads to the conclusion that the getuptime was neces-
        sarily after this long valley and any peaks or valleys
        before it may be removed.

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
        
        # Valleys that are too long are classified as invalid.
        peaks_and_valleys["valid"] = True
        peaks_and_valleys.loc[peaks_and_valleys["class"] == "v","valid"] = peaks_and_valleys.loc[peaks_and_valleys["class"] == "v","length"] < self.getuptime_high_probability_sleep_valley_length

        if self.do_remove_before_long_valley:
            invalid_regions = peaks_and_valleys[peaks_and_valleys["valid"] == False].index.to_numpy()
            invalid_region_count = len(invalid_regions)

            if invalid_region_count > 0:
                if invalid_regions[invalid_region_count-1] < peaks_and_valleys_count-2:
                    peaks_and_valleys = peaks_and_valleys.loc[invalid_regions[invalid_region_count-1]:,:]

            if self.verbose:
                print("remove long valley\n",peaks_and_valleys)

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

        # Peaks and valleys will be analyzed as if the last region
        # (closer to awake state) was the first. This is done so 
        # that this refinement step mirror that of bedtime refine-
        # ment.
        last_index = peaks_and_valleys_count-1

        # print(peaks_and_valleys)

        if (peaks_and_valleys_count > 2):
            # First region is treated separately and may be remo- 
            # ved, if it's a valley.
            if (peaks_and_valleys.at[last_index,"class"] == "v"):
                if (peaks_and_valleys.at[last_index,"length"] < self.getuptime_minimum_valley_length):
                    # In this case, the first valley is merged into
                    # it's succeeding peak
                    peaks_and_valleys = remove_peak_valley(peaks_and_valleys,last_index,self.refinement_window_activity,self.metric)
                    peaks_and_valleys_count = len(peaks_and_valleys)
                    last_index = peaks_and_valleys_count-1

                else:
                    # Specific criteria are applied if the first
                    # valley has a valid length.
                    if (peaks_and_valleys_count == 3):
                        # If there are only 3 regions, in this ca-
                        # se it's valley-peak-valley. If the center
                        # peak is relatively shorter than the follo-
                        # wing valley or if that is a long valley,
                        # all 3 will merge into a single valley.
                        if (
                            (peaks_and_valleys.at[last_index-2,"length"] > 60)
                            or (4*peaks_and_valleys.at[last_index-1,"length"] < peaks_and_valleys.at[last_index-2,"length"])
                            ):
                            peaks_and_valleys = remove_peak_valley(peaks_and_valleys,last_index-1,self.refinement_window_activity,self.metric)
                            peaks_and_valleys_count = len(peaks_and_valleys)
                            last_index = peaks_and_valleys_count-1


                            ### probable error
                            peaks_and_valleys.at[last_index,"class"] = "p"


                    else:
                        # If there's more than 3 regions, the first
                        # valley will only be removed if it's median
                        # activity is significantly lower than that
                        # of its following peak given that it has a
                        # minimum length.
                        if (
                            (peaks_and_valleys.at[last_index-1,"length"] >= 20)
                            and (3*peaks_and_valleys.at[last_index,"median"] < peaks_and_valleys.at[last_index-1,"median"])
                            ):
                            peaks_and_valleys = remove_peak_valley(peaks_and_valleys,last_index,self.refinement_window_activity,self.metric)
                            peaks_and_valleys_count = len(peaks_and_valleys)
                            last_index = peaks_and_valleys_count-1


            if (peaks_and_valleys_count > 3):
                # t iterates through the index in reverse order, as
                # explained above, to mirror the bedtime refinement.
                t = 1
                while (t < peaks_and_valleys_count) and ((peaks_and_valleys_count > 2) and (peaks_and_valleys_count != 3)):
                    remove = False
                    # Remove conditions here are very similar to their 
                    # bedtime counterparts.

                    # All regions are subjected to length-based threshol-
                    # ding and, if not filtered, they are evaluated ba-
                    # sed off other criteria
                    if (peaks_and_valleys.at[last_index-t,"class"] == "p"):
                        if (peaks_and_valleys.at[last_index-t,"length"] < self.getuptime_minimum_peak_length):
                            if t > 1:
                                if t < peaks_and_valleys_count-2:
                                    remove = True
                                
                                else:
                                    # If an invalid peak is the penultima-
                                    # te region, it'll be removed only if 
                                    # the preceding valey has a significa-
                                    # tive length and low level of activi-
                                    # ty (based on a combination of crite-
                                    # ria).
                                    if (
                                            (peaks_and_valleys.at[last_index-t,"above_threshold_proportion"] < 0.8)  # Recently added
                                            and (
                                                 (peaks_and_valleys.at[last_index-t+1,"length"] >= 70)
                                                 or (
                                                     (peaks_and_valleys.at[last_index-t+1,"length"] > self.getuptime_minimum_valley_length)
                                                     and (
                                                           (peaks_and_valleys.at[last_index-t+1,"zero_proportion"] > 0.6)
                                                           or (
                                                               (peaks_and_valleys.at[last_index-t+1,"above_threshold_proportion"] < 0.25)
                                                                and (peaks_and_valleys.at[last_index-t+1,"zero_proportion"] > 0.4)
                                                              )
                                                          )
                                                    )
                                                )
                                        ):
                                        remove = True

                            else:
                                # If an invalid peak is the second region,
                                # it'll be removed only if the preceding val-
                                # ley has a significative length and propor-
                                # tion of zeros.
                                if (
                                    (peaks_and_valleys.at[last_index-t+1,"length"] >= 30) and 
                                    (peaks_and_valleys.at[last_index-t+1,"zero_proportion"] > 0.6)
                                   ):
                                    remove = True

                    else:#(peaks_and_valleys.at[t,"class"] == "v")
                        if (t < peaks_and_valleys_count-1):
                            if (peaks_and_valleys.at[last_index-t,"length"] < self.getuptime_minimum_valley_length):
                                if t > 1:
                                    remove = True
                                
                                else:
                                # If a valley with an invalid length is the
                                # second region, it will be removed only if 
                                # it has a low proportion of zeros. 
                                    if (
                                        #(peaks_and_valleys.at[last_index-t,"zero_proportion"] < 0.1)
                                        (peaks_and_valleys.at[last_index-t,"above_fixed_threshold_proportion"] > 0.1)
                                        and (peaks_and_valleys.at[last_index-t-1,"zero_proportion"] < 0.1) # Recently added
                                        ):
                                        remove = True

                            else:
                            # Valleys with valid lengths may be removed if 
                            # their features are not good. This is evalua-
                            # ted by several criteria combinations.
                                remove_points = 0
                                if (peaks_and_valleys.at[last_index-t,"above_threshold_proportion"] > 0.1):
                                    remove_points += 1

                                if (peaks_and_valleys.at[last_index-t,"zero_proportion"] < 0.25):
                                    remove_points += 1

                                if (peaks_and_valleys.at[last_index-t,"mean"] >= 0.66*self.metric):
                                    remove_points += 0.5

                                if (
                                    (peaks_and_valleys.at[last_index-t-1,"length"] > peaks_and_valleys.at[last_index-t,"length"])
                                   and (peaks_and_valleys.at[last_index-t-1,"mean"] > 3*peaks_and_valleys.at[last_index-t,"mean"])
                                   ):
                                    remove_points += 1

                                if (
                                    (peaks_and_valleys.at[last_index-t+1,"length"] > 2*peaks_and_valleys.at[last_index-t,"length"])
                                   and (peaks_and_valleys.at[last_index-t+1,"mean"] > 3*peaks_and_valleys.at[last_index-t,"mean"])
                                   ):
                                    remove_points += 1

                                if (
                                    (peaks_and_valleys.at[last_index-t,"length"]/len(self.refinement_window_activity) >= 0.3)
                                    # or (peaks_and_valleys.at[t,"length"] >= 1.5*peaks_and_valleys.at[t+1,"length"])
                                   ):
                                    remove_points -= 1

                                if remove_points > 2:
                                    remove = True


                    if remove:
                        # When a valley or a peak is removed, it's
                        # actually merged into the neighboring re-
                        # gions to create a large new region com-
                        # posed by the 3 regions. 
                        peaks_and_valleys = remove_peak_valley(peaks_and_valleys,last_index-t,self.refinement_window_activity,self.metric)
                        peaks_and_valleys_count = len(peaks_and_valleys)
                        last_index = peaks_and_valleys_count-1

                        if self.verbose:
                            print("t",t,"peaks_and_valleys\n",peaks_and_valleys)

                    else:
                        t += 1

        peaks_and_valleys_count = len(peaks_and_valleys)
        peaks_and_valleys.index = range(peaks_and_valleys_count)
        return peaks_and_valleys


    def identify_getuptime_candidates(self,
                                      peaks_and_valleys,
                                      ):
        """In the context of a getuptime refinement, getuptime 
        candidates will be the first epochs of the peak regions.
        Also, if the last region is a valley, it's last epoch 
        will also be a candidate.

        Parameters
        ----------
        peaks_and_valleys: pd.DataFrame
                Contains the information of the location and fea-
                tures of peaks and valleys inside a refinement
                window.

        Returns
        -------
        getuptime_candidates: np.array [int]
                Indexes of the getuptime candidate epochs.
        """

        peaks_and_valleys_count = len(peaks_and_valleys)
        getuptime_candidates = []
        for t in range(peaks_and_valleys_count):
            start = int(self.refinement_window_start + peaks_and_valleys.at[t,"start"])
            end = int(self.refinement_window_start + peaks_and_valleys.at[t,"end"])
            self.refinement_window_levels[start:end] = peaks_and_valleys.at[t,"mean"]

            if (peaks_and_valleys.at[t,"class"] == "p"):
                if t > 0:
                    new = int(peaks_and_valleys.at[t,"start"])
                    getuptime_candidates.append(new)

        if (peaks_and_valleys.at[peaks_and_valleys_count-1,"class"] == "v"):
            new = int(peaks_and_valleys.at[peaks_and_valleys_count-1,"end"])
            if self.refinement_window_start+new == self.data_length:
                new -=1
            getuptime_candidates.append(new)

        return getuptime_candidates


    def choose_best_getuptime_candidate(self,
                                        getuptime_candidates,
                                        short_window_filter_metric_crossing_up,
                                        ):
        """Assigns scores to the getuptime candidates based on their
        features and location in order to determine which is one is 
        most likely the actual transition between asleep and awake 
        states.

        Parameters
        ----------
        getuptime_candidates: np.array [int]
                Indexes of the getuptime candidate epochs.
        short_window_filter_metric_crossing_up: float
                Indexes of the epochs where the short-window-me-
                dian-filtered activity goes higher than the me-
                tric.

        Returns
        -------
        refined_getuptime : int
                Index of the refined getuptime epoch
        """

        getuptime_candidates_count = len(getuptime_candidates)
        if getuptime_candidates_count > 0:
            getuptime_candidate_scores = pd.DataFrame(getuptime_candidates,columns=["candidate"])
            getuptime_candidate_scores["datetime_stamp"] = [self.datetime_stamps[self.refinement_window_start+getuptime_candidates[candidate]] for candidate in range(getuptime_candidates_count)]
            
            # The presence of a activity median difference peak
            # near a candidate is indicative of a sudden rise in 
            # the level of activity, this is associated with the 
            # transition from asleep to awake.
            getuptime_candidate_scores["activity_median_difference"] = [get_peak(self.refinement_window_activity_median_difference_smoothed,getuptime_candidates[candidate]) for candidate in range(getuptime_candidates_count)]
            
            # In the context of getuptime detection, the best can-
            # didates have low numbers of epochs with activity a-
            # bove metric.
            getuptime_candidate_scores["epochs_below_metric_after"] = self.after_candidate_window

            # If the number of epochs with activity above metric
            # is below the user-set threshold, this will be True.
            getuptime_candidate_scores["thresholded"] = False

            # This indicates if there's any invalid gaps near the
            # candidate. Before evaluation, all are initialized 
            # as invalid.
            getuptime_candidate_scores["gap_after"] = True

            # As features are evaluated, the candidates will be
            # scored.
            getuptime_candidate_scores["score"] = 0.0

            # Being the first (relative to datetime) is an interes-
            # ting feature for the candidate, so it may be scored.
            if self.getuptime_score_first_candidate:
                getuptime_candidate_scores.at[0,"score"] += self.getuptime_first_candidate_score

            if self.verbose:
                print("getuptime_candidate_scores\n",getuptime_candidate_scores)

            # The candidate with the most abrupt peak in median
            # activity difference gets an extra score.
            getuptime_candidate_scores.sort_values(by=["activity_median_difference","candidate"],axis=0,ascending=[False,True],ignore_index=True,inplace=True)
            getuptime_candidate_scores.sort_values(by=["activity_median_difference","candidate"],axis=0,ascending=False,ignore_index=True,inplace=True)
            getuptime_candidate_scores.at[0,"score"] += self.getuptime_best_median_difference_candidate_score
            if self.verbose:
                print("getuptime_candidate_scores\n",getuptime_candidate_scores)


            # The candidate that's closest to the first short-win-
            # down-median-filtered activity metric crossing-up 
            # gets an extra score.
            if len(short_window_filter_metric_crossing_up) > 0:   
                short_window_filter_metric_crossing_up = short_window_filter_metric_crossing_up[0]
                metric_crossing_distance = np.absolute(getuptime_candidates - short_window_filter_metric_crossing_up)
                best_crossing_distance_candidate = np.argmin(metric_crossing_distance)

                getuptime_candidate_scores.loc[getuptime_candidate_scores["candidate"]==getuptime_candidates[best_crossing_distance_candidate],"score"] += self.getuptime_best_crossing_distance_candidate_score
                if self.verbose:
                    print("short_window_filter_metric_crossing_up",short_window_filter_metric_crossing_up)
                    print("best_crossing_distance_candidate",best_crossing_distance_candidate)
                    print("getuptime_candidate_scores\n",getuptime_candidate_scores)
            
            # Lastly, the neighborhood of each candidate is evalu-
            # ated and scores are given based on their number of
            # following epochs above metric
            candidate_index = 0
            while (candidate_index < getuptime_candidates_count):
                candidate = self.refinement_window_start + getuptime_candidates[candidate_index]

                after_candidate_window_end = candidate+self.after_candidate_window
                if after_candidate_window_end > self.data_length:
                    after_candidate_window_end = self.data_length
                activity_after_candidate = self.activity[candidate:after_candidate_window_end]   # We'll look at a window after the candidate epoch

                valid_gap = True
                g = candidate
                while valid_gap and (g+1 < after_candidate_window_end):
                    valid_gap,datetime_seconds_gap = self.datetime_gap_check(g,direction="forward",return_gap=True)
                    if not valid_gap:
                        if self.verbose:
                            print("gap",datetime_seconds_gap)
                    g += 1
           
                if valid_gap:
                    getuptime_candidate_scores.loc[getuptime_candidate_scores["candidate"] == getuptime_candidates[candidate_index],"gap_after"] = False

                    epochs_below_metric_after = np.where(activity_after_candidate <= self.metric,1,0)   # And evaluate how many epochs inside the window are above the metric
                    epochs_below_metric_after = np.sum(epochs_below_metric_after)

                    getuptime_candidate_scores.loc[getuptime_candidate_scores["candidate"] == getuptime_candidates[candidate_index],"epochs_below_metric_after"] = epochs_below_metric_after
                    if epochs_below_metric_after <= self.getuptime_maximum_epochs_above_metric_after_candidate:
                        getuptime_candidate_scores.loc[getuptime_candidate_scores["candidate"] == getuptime_candidates[candidate_index],"thresholded"] = True

                candidate_index += 1

            thresholded_getuptime_candidates = getuptime_candidate_scores[getuptime_candidate_scores["thresholded"]]
            if self.verbose:
                print("getuptime_candidate_scores\n",getuptime_candidate_scores)
                print("thresholded_getuptime_candidates\n",thresholded_getuptime_candidates)

            if len(getuptime_candidate_scores[getuptime_candidate_scores["gap_after"]==False]) > 0:
                if len(thresholded_getuptime_candidates) == 0:   # If there aren't any valid candidates, the one with the least above-metric epochs is chosen
                    getuptime_candidate_scores.sort_values(by=["gap_after","epochs_below_metric_after",],axis=0,ignore_index=True,inplace=True)
                    getuptime_candidate_scores.at[0,"score"] += self.getuptime_best_epochs_above_metric_after_score

                else:
                    if self.getuptime_maximum_epochs_above_metric_after_candidate > 0:
                        thresholded_score_factor = (-self.getuptime_thresholded_candidate_score_amplitude/self.getuptime_maximum_epochs_above_metric_after_candidate)
                    else:
                        thresholded_score_factor = (-self.getuptime_thresholded_candidate_score_amplitude/1e-3)
                    
                    thresholded_getuptime_candidates.loc[:,"score"] = thresholded_score_factor*(thresholded_getuptime_candidates.loc[:,"epochs_below_metric_after"]-self.getuptime_maximum_epochs_above_metric_after_candidate)
                    if self.verbose:
                        print("scored thresholded_getuptime_candidates\n",thresholded_getuptime_candidates)

                    getuptime_candidate_scores.loc[getuptime_candidate_scores["thresholded"],"score"] += thresholded_getuptime_candidates["score"]+self.getuptime_thresholded_candidate_score_minimum

            if self.verbose:
                print("getuptime_candidate_scores\n",getuptime_candidate_scores)

            getuptime_candidate_scores.sort_values(by=["score","candidate"],axis=0,ascending=False,ignore_index=True,inplace=True)
            if self.verbose:
                print("getuptime_candidate_scores\n",getuptime_candidate_scores)

            # The highest scored candidate is chosen as the refi-
            # ned getuptimes.
            refined_getuptime = self.refinement_window_start + int(getuptime_candidate_scores.at[0,"candidate"])
            if self.verbose:
                print("getuptime_candidate_scores\n",getuptime_candidate_scores)
                print("top_grader",getuptime_candidate_scores.at[0,"candidate"])

        else:
            refined_getuptime = self.initial_transition_candidate

        if self.verbose:
            print("refined_getuptime",refined_getuptime,self.datetime_stamps[refined_getuptime])
            # input()

        return refined_getuptime

    def refine(self,
               refinement_window_levels,
               initial_transition_candidate,
               previous_transition,
               next_transition,
               verbose=False
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
        refined_getuptime : int
                Index of the refined getuptime epoch
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
            print("\n\ngetup time refinement")
            print("initial",self.datetime_stamps[self.initial_transition_candidate])

        initial_candidate = self.initial_transition_candidate+1
        refinement_window_end = self.compute_refinement_window_end(initial_candidate)
        if self.verbose:
            print("end0",self.datetime_stamps[refinement_window_end])

        initial_candidate =  self.initial_transition_candidate-1
        refinement_window_start = self.compute_refinement_window_start(initial_candidate,refinement_window_end)     
        if self.verbose:
            print("start0",self.datetime_stamps[refinement_window_start])

        valid_gap = self.datetime_gap_check(refinement_window_start)
        if not valid_gap:
            refinement_window_start,refinement_window_end = self.bridge_gap_validation(refinement_window_start)

        refinement_window_datetime_difference = datetime_diff(self.datetime_stamps[refinement_window_start:refinement_window_end+1])
        refinement_window_datetime_gaps = np.where(refinement_window_datetime_difference >= self.maximum_allowed_gap,1,0)
        if np.sum(refinement_window_datetime_gaps) > 0:
            first_gap_location = list(refinement_window_datetime_gaps).index(1)
            if first_gap_location > len(refinement_window_datetime_difference)-first_gap_location:
                refinement_window_end = refinement_window_start + first_gap_location - 1
            else:
                refinement_window_start = refinement_window_start + first_gap_location + 1

        if refinement_window_start >= refinement_window_end:
            refinement_window_end = refinement_window_start + 60
            if refinement_window_end >= self.data_length:
                refinement_window_end = self.data_length-1

        self.refinement_window_start = refinement_window_start
        self.refinement_window_end = refinement_window_end
        self.refinement_window_activity_median = self.compute_refinement_window_median(self.refinement_window_start,self.refinement_window_end)

        self.metric = self.compute_metric(self.refinement_window_activity_median)     

        self.refinement_window_activity_median_difference = np.diff(self.refinement_window_activity_median)
        self.refinement_window_activity_median_difference_smoothed = median_filter(self.refinement_window_activity_median_difference,self.median_filter_half_window_size,)

        self.refinement_window_activity = self.activity[refinement_window_start:refinement_window_end+1]
        peaks_and_valleys = identify_peaks_and_valleys(self.refinement_window_activity,self.refinement_window_activity_median,self.metric)
        peaks_and_valleys["above_fixed_threshold_proportion"] = 0.0
        for region_index in range(len(peaks_and_valleys)):
                start = peaks_and_valleys.at[region_index,"start"]
                end = peaks_and_valleys.at[region_index,"end"]
                region_activity = self.refinement_window_activity[start:end]
                peaks_and_valleys.at[region_index,"above_fixed_threshold_proportion"] = sum(np.where(region_activity > 10,1,0))/(end-start)

        if self.verbose:
            print("metric",self.metric)
            print("start",self.datetime_stamps[self.refinement_window_start])
            print("end",self.datetime_stamps[self.refinement_window_end])
            print("peaks_and_valleys\n",peaks_and_valleys)

        peaks_and_valleys = self.remove_after_long_tall_peak(peaks_and_valleys)

        peaks_and_valleys = self.remove_before_long_valley(peaks_and_valleys)

        self.peaks_and_valleys = self.filter_peaks_and_valleys(peaks_and_valleys)        
        if self.verbose:
            print("filtered peaks_and_valleys\n",peaks_and_valleys)

        peaks_and_valleys_count = len(self.peaks_and_valleys)
        
        if self.verbose:
                print("initial borders")
                print(self.refinement_window_start)
                print(self.refinement_window_end)
                print(self.peaks_and_valleys)

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

        getuptime_candidates = self.identify_getuptime_candidates(self.peaks_and_valleys)                    
        if self.verbose:
            print("getuptime_candidates",getuptime_candidates)

        short_window_filter_metric_crossing = np.where(self.short_window_activity_median[self.refinement_window_start:self.refinement_window_end+1] >= self.metric, 1, 0)
        short_window_filter_metric_crossing = np.diff(np.concatenate(([0],short_window_filter_metric_crossing)))
        short_window_filter_metric_crossing_up = (short_window_filter_metric_crossing > 0).nonzero()[0]

        self.refined_getuptime = self.choose_best_getuptime_candidate(getuptime_candidates,short_window_filter_metric_crossing_up)

        return self.refined_getuptime,self.refinement_window_start,self.refinement_window_end,self.refinement_window_activity_median,self.refinement_window_activity_median_difference_smoothed,refinement_window_levels,self.metric