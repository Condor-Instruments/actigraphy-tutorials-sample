"""Parameter tuning and analysis studies for bimodal offwrist algorithm 

Author: Julius A. P. P. de Paula (--/2023)
"""

import numpy as np
from scipy.ndimage import binary_closing, binary_opening
from scipy.signal import butter, filtfilt, freqz, find_peaks, peak_prominences
from scipy import stats
import matplotlib.pyplot as plt
from datetime import timedelta
import time as ttime

import sys,inspect,os

from describe_offwrist_periods import describe_offwrist_periods
from bimodal_thresh_development import bimodal_thresh
from functions import *

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


class BimodalOffwristRefiner:
    def __init__(self,
                 ashman_d_minimum=1.5,ashman_d_maximum=2.6,
                 temperature_variance_threshold_quantile=0.9,offwrist_minimum_low_temperature_proportion=0.6,
                 minimum_preceding_onwrist_period_length=40,minimum_onwrist_length=20,minimum_offwrist_length=15,search_border_configuration="mod",
                 lumus_file_minimum_activity_median_high_proportion=0.25,maximum_low_temperature_variance_proportion_around_border=0.3,
                 activity_threshold_quantile=0.35,low_activity_threshold=500,minimum_low_activity_proportion=0.8,
                 do_zero_activity_filter=False,do_onwrist_zero_activity_proportion_filter=False,do_onwrist_length_filter=True,too_close_offwrist_low_activity_criteria=True,do_offwrist_length_filter=True,do_low_temperature_variance_filter=False,
                 quick_delete_start=False,centered_low_temperature_variance_proportion=False,do_low_activity_filter=True,do_sleep_low_temperature_filter=False,
                 do_description_report_based_filter=True,do_temperature_difference_filter=False,do_capacitive_sensor_variance_filter=True,do_sleep_filter=True,
                 sleep_activity_filter_half_window_length=120,sleep_low_activity_threshold_configuration="both",sleep_all_activity_quantile=0.4,sleep_positive_activity_quantile=0.05,maximum_offwrist_sleep_proportion=0.4,do_short_sleep_filter=True,
                 do_report_activity_around_filter=True,do_report_border_activity_filter=True,do_report_border_concentration_filter=True,
                 do_report_zero_activity_proportion_filter=True,report_zero_activity_proportion_minimum=0.35,border_concentratition_minimum=0.5,report_activity_around_minimum=0.1,
                 do_statistical_measure_filter=False,

                 do_analyze_sleep_borders=False,
                 do_valley_peak_algorithm=True,
                 valley_quantile=0.97,
                 peak_quantile=0.97,
                 temperature_threshold_refinement_intensity=0.5,
                 temperature_threshold_refinement_quantile=0.6,
                 do_temperature_threshold_refinement=False,
                 do_report_low_activity_proportion_filter=True,
                 report_low_activity_proportion_minimum=0.5,
                 do_report_low_activity_after=True,
                 
                 do_forbidden_zone=False,
                 possible_window_hours=1,

                 do_valley_peak_filter=False,
                 bimodal_maximum_offwrist_proportion=0.4,
                 bimodal_minimum_low_activity_proportion=0.7,
                 do_temperature_criteria=False,
                 do_short_low_temperature_filter=True,
                 sleep_low_temperature_proportion_maximum=0.333,
                 
                 offwrist_maximum_temperature_difference_median=1.0,
                 offwrist_minimum_temperature_difference_median=0.75,
                 offwrist_minimum_capsensor1_median=165,
                 offwrist_minimum_capsensor2_median=120,
                 offwrist_maximum_capsensor1_std=165,
                 offwrist_maximum_capsensor2_std=120,

                 half_day_length_validation=False,
                 decrease_ratio_minimum=0.0,
                 next_possible_length_criteria=True,
                 short_valley_peak_offwrist_criteria = True,
                 medium_valley_peak_offwrist_criteria = True,
                 valley_peak_low_temperature_minimum = 0.5,
                 short_offwrist_minimum_decrease_ratio = 1.25,
                 lumus_short_offwrist_maximum_capsensor_pvalue = 5e-4,
                 trust_short_offwrist_maximum_temperature_difference_pvalue = 5e-4,
                 short_offwrist_length = 20,
                 offwrists_need_decreasing_temperature=True,
                 skip_description_filters=True,
                 ):
        """Improves an initial offwrist detection using read information. The 
        improvements are achieved by 1º: applying a sequence of filters that 
        delete invalid periods; 2º: trying and finding better borders for each 
        period and 3º: applying another sequence of filters.


        Parameters
        ----------

        *** First stage parameters ***
        minimum_onwrist_length : int
                Onwrist periods with a length shorter than this number may be 
                filtered out.
        do_onwrist_length_filter : boolean
                If True, onwrist periods that are too short will be filtered
                out.
        do_onwrist_zero_activity_proportion_filter : boolean
                If True, onwrist periods with a high zero activity proportion
                will be filtered out.
        

        *** Second stage parameters ***
        temperature_variance_threshold_quantile : float
                Quantile used to stablish a threshold for "high" temperature
                variance.
        minimum_preceding_onwrist_period_length : int
                Offwrist periods surrounding an onwrist period with a length
                shorter than this parameter may be merged to one.
        search_border_configuration : {"disc","mod","mod2"}
                Defines how offwrist period borders will evaluated to determine
                if they are valid
        maximum_low_temperature_variance_proportion_around_border : float
                Offwrist periods must have high temperature variance levels clo-
                se to its borders. This is the maximum allowed proportion of
                low values
        activity_threshold_quantile : float
                Quantile used to stablish a threshold for "low" activity
        low_activity_threshold : float
                Stablishes the level of activity that is considered typical in-
                side offwrist periods
        too_close_offwrist_low_activity_criteria : boolean
                If True, offwrist periods that have a short onwrist period bet-
                ween them, to be combined to one, that onwrist period must al-
                so meet a low activity criteria
        quick_delete_start : boolean
                If True, offwrist border candidates that don't meet the basic
                temperature variance are filtered out without further analysis
        centered_low_temperature_variance_proportion : boolean
                Defines if the low temperature variance proportion around a 
                border candidate will be calculated using a centered window or 
                using only past or future epochs depending on the case.


        *** Third stage parameters ***
        minimum_low_activity_proportion : float
                Valid offwrist periods must have a high proportion of low acti-
                vity values. This is the minimum necessary proportion.
        do_low_activity_filter : boolean
                If True, offwrist periods that don't have enough low activity
                proportion get filtered out
        do_zero_activity_filter : boolean
                If True, offwrist periods that don't have enough zero activity
                proportion get filtered out
        offwrist_minimum_low_temperature_proportion : float
                Valid offwrist periods must have a high proportion of low tem-
                erature values. This is the minimum necessary proportion.
        do_low_temperature_variance_filter : boolean
                If True, offwrist periods that don't have high temperature va-
                riance levels on its borders get filtered out
        minimum_offwrist_length : int
                Offwrist periods with a length shorter than this number may be 
                filtered out.
        do_offwrist_length_filter : boolean
                If True, offwrist periods that are too short get filtered out
        do_capacitive_sensor_variance_filter : boolean
                If True, during sleep-estimate-filter, offwrist periods that 
                don't meet the sleep proportion criteria may be spared if they
                have a sufficiently low capacitive sensor difference variance
        do_sleep_filter : boolean
                If True, offwrist periods that have too much overlap with the
                estimated sleep periods will be filtered out
        sleep_activity_filter_half_window_length : int
                Window length used in the sleep estimation activity median fil-
                ter
        sleep_low_activity_threshold_configuration : {"all","zp","both"}
                If "all": sleep low activity threshold will be computed u-
                sing a quantile of all valid activity data 
                If "zp": sleep low activity threshold will be computed u-
                sing a quantile of the positive valid activity data
                If "both": a combination of the previous strategies is u-
                sed
        sleep_all_activity_quantile : float
                Quantile used when sleep_low_activity_threshold_configurati-
                on==all (also when ==both)
        sleep_positive_activity_quantile : float
                Quantile used when sleep_low_activity_threshold_configurati-
                on==zp (also when ==both)
        do_sleep_low_temperature_filter : boolean
                If True, estimated sleep periods will have its borders refined
                to meet high temperature variance criteria
        do_short_sleep_filter : boolean
                If True, estimated sleep periods that are too short will be
                filtered out
        maximum_offwrist_sleep_proportion : float
                Maximum allowed overlap between offwrist periods and estimated
                sleep information
        ashman_d_minimum : float
                Minimum value of the Ashman's D statistic for a given set of 
                offwrist information to be considered bimodal (offwrist and 
                onwrist epochs present).
        ashman_d_maximum : float
                Maximum value of the Ashman's D statistic for a given set of 
                offwrist information to be considered bimodal (offwrist and 
                onwrist epochs present).
        do_description_report_based_filter : boolean
                If True, refined offwrist periods that don't meet a series of 
                feature-based criteria will be filtered out
        do_report_border_concentration_filter : boolean
                If True, offwrist periods that don't meet activity and tempe-
                rature border concentration criteria will be filtered out
        report_zero_activity_proportion_minimum : float
                Offwrist periods that don't have at least this proportion of
                zero values may be filtered out
        do_report_zero_activity_proportion_filter : boolean
                If True, offwrist periods that don't have enough zeros will be
                filtered out
        do_report_activity_around_filter : boolean
                If True, offwrist periods don't have enough activity next to 
                their borders will be filtered out
        border_concentratition_minimum : float
                Minimum concentration measure for both activity and temperatu-
                re variance for the report_border_activity_filter
        report_activity_around_minimum : float
                Minimum activity next to the offwrist border measure for the 
                report_border_activity_filter
        do_report_border_activity_filter : boolean
                If True, offwrist periods that don't meet simultaneous activi-
                ty concentration, temperature variance concentration and acti-
                vity next to the borders criteria will be filtered out
        do_temperature_difference_filter : boolean
                If True, offwrist periods that have too much temperature diffe-
                rence variance will be filtered out
        lumus_file_minimum_activity_median_high_proportion : float
                Minimum value for the proportion of high values of median acti-
                vity. ActLumus-read files that meet this criteria may be spared
                from previous filters
        do_statistical_measure_filter : boolean
                If True, offwrist periods invalidated by previous criteria may be
                spared if they meet new criteria based on the statistical mea-
                sures obtained from the additional information

                
        do_analyze_sleep_borders : boolean
                If True, the sleep border algorithm will take place. It compu-
                tes statistical features surrounding the borders of estimated
                sleep periods.

                
        do_valley_peak_algorithm : boolean
                If True, the valley-peak-offwrist algorithm will take place.
                By using the temperature derivative most intense (daily) val
                -leys and peaks, it is possible to locate regions where the 
                temperature rapidly diminishes and then rapidly increases, 
                which are indicative of the onset and offset of an offwrist 
                period.
        valley_quantile : float
                Temperature derivative valleys below this quantile threshold
                will be listed as possible offwrist onsets.
        peak_quantile : float
                Temperature derivative peaks above this quantile threshold
                will be listed as possible offwrist offsets.

                
        temperature_threshold_refinement_intensity : float
                Factor used to correct the temperature threshold, bringing 
                it to higher values.
        temperature_threshold_refinement_quantile : float
                Quantile used to stablish a threshold for "high" temperature,
                related to sleep.
        do_temperature_threshold_refinement : boolean
                If True, the temperature threshold will be updated by consi-
                dering the current predicted offwrist.

        do_report_low_activity_proportion_filter : boolean
                If True, during the description-report-based filtering part
                on the 3º filtering stage, offwrist periods that have a low
                proportion of low activity epochs will be excluded from the
                detection. Works alongside do_report_low_activity_after.
        report_low_activity_proportion_minimum : float
                Related to the previously presented filter. Minimum propor-
                tion of low-activity epochs an offwrist period candidate must
                have to be accepted.
        do_report_low_activity_after : boolean
                If True, during the description-report-based filtering part
                on the 3º filtering stage, offwrist periods that have a low
                proportion of low activity epochs or a low proportion of zero
                activity epochs will be excluded from the detection. Works
                alongside do_report_low_activity_proportion_filter.

        do_forbidden_zone : boolean
                If True, the forbidden zones algorithm will take place. It
                defines zones in the center of estimated sleep periods as
                forbidden, meaning that offwrist period candidates inside
                those zones will ye excluded.

        do_valley_peak_filter : boolean
                If True, offwrist periods detected in the stages prior to the
                application of the valley-peak algorithm will be filtered with
                respect to whether or not they are overlapped by valley-peak
                detections.
                
        bimodal_maximum_offwrist_proportion : float
                If True, input time series with an initial proportion of esti-
                mated offwrist epochs larger than this threshold will likely be 
                categorized as unimodal.
        bimodal_minimum_low_activity_proportion : float
                If True, input time series with a proportion of low positive 
                activity epochs larger than this threshold will likely be cate-
                gorized as unimodal.
        do_temperature_criteria : boolean
                If True, input time series with a proportion of low positive 
                activity epochs larger than a specific threshold will likely 
                be categorized as unimodal.

        do_short_low_temperature_filter : boolean
                If True, estimated sleep periods with a proportion of low tem-
                perature epochs larger than sleep_low_temperature_proportion_
                _maximum will be filtered.
        sleep_low_temperature_proportion_maximum : float
                Threshold from previously described boolean.

        offwrist_maximum_temperature_difference_median : float
                Only for ActTrust inputs. Statistitically-based maximum allow-
                ed temperature difference median for an offwrist candidate in 
                the 3º refinement stage.
        offwrist_minimum_temperature_difference_median : float
                Only for ActTrust inputs. Statistitically-based minimum tempe-
                rature difference median level needed for a valid offwrist can-
                didate in the description-report-based  filter in the 3º refi-
                nement stage.
        offwrist_minimum_capsensor1_median : float
                Only for ActLumus inputs. Statistitically-based minimum capaci-
                tive sensor 1 median level needed for a valid offwrist candida-
                te in the 3º refinement stage.
        offwrist_minimum_capsensor2_median : float
                Only for ActLumus inputs. Statistitically-based minimum capaci-
                tive sensor 2 median level needed for a valid offwrist candida-
                te in the 3º refinement stage.
        offwrist_maximum_capsensor1_std : float
                Only for ActLumus inputs. Statistitically-based maximum allow-
                ed capacitive sensor 1 standard deviation level for a valid off-
                wrist candidate in the 3º refinement stage.
        offwrist_maximum_capsensor2_std : float
                Only for ActLumus inputs. Statistitically-based maximum allow-
                ed capacitive sensor 2 standard deviation level for a valid off-
                wrist candidate in the 3º refinement stage.


        half_day_length_validation : boolean
                If True, long offwrists with duration of at least 12 hours will
                not be filtered in some operations.
        decrease_ratio_minimum : float
                Minimum temperature derivative decrease ratio needed for an off-
                wrist to be valid.
        next_possible_length_criteria : boolean
                If True, during the valley-peak offwrist detection, in sequences
                with valley-valley-peak configuration, if the second valley is 
                closer than minimum offwrist length to the peak, the algorithm
                will create a candidate using the first valley and the peak.
        short_valley_peak_offwrist_criteria : boolean
                If True, during the valley-peak offwrist detection, a specific
                criteria will be applied to 
        medium_valley_peak_offwrist_criteria : boolean
                If True, during the valley-peak offwrist detection,
        valley_peak_low_temperature_minimum : float

        short_offwrist_minimum_decrease_ratio : float

        lumus_short_offwrist_maximum_capsensor_pvalue : float

        short_offwrist_length : float

        offwrists_need_decreasing_temperature : boolean

        skip_description_filters : boolean

        """

        self.do_onwrist_length_filter = do_onwrist_length_filter
        self.minimum_onwrist_length = minimum_onwrist_length
        self.do_onwrist_zero_activity_proportion_filter = do_onwrist_zero_activity_proportion_filter

        self.temperature_variance_threshold_quantile = temperature_variance_threshold_quantile
        self.minimum_preceding_onwrist_period_length = minimum_preceding_onwrist_period_length
        self.search_border_configuration = search_border_configuration
        self.maximum_low_temperature_variance_proportion_around_border = maximum_low_temperature_variance_proportion_around_border
        self.activity_threshold_quantile = activity_threshold_quantile
        self.low_activity_threshold = low_activity_threshold
        self.too_close_offwrist_low_activity_criteria = too_close_offwrist_low_activity_criteria
        self.quick_delete_start = quick_delete_start
        self.centered_low_temperature_variance_proportion = centered_low_temperature_variance_proportion
        
        self.do_low_activity_filter = do_low_activity_filter
        self.minimum_low_activity_proportion = minimum_low_activity_proportion
        self.do_zero_activity_filter = do_zero_activity_filter
        self.do_low_temperature_variance_filter = do_low_temperature_variance_filter
        self.offwrist_minimum_low_temperature_proportion = offwrist_minimum_low_temperature_proportion
        self.do_offwrist_length_filter = do_offwrist_length_filter
        self.minimum_offwrist_length = minimum_offwrist_length
        self.do_capacitive_sensor_variance_filter = do_capacitive_sensor_variance_filter
        self.do_sleep_filter = do_sleep_filter
        self.sleep_activity_filter_half_window_length = sleep_activity_filter_half_window_length
        self.sleep_low_activity_threshold_configuration = sleep_low_activity_threshold_configuration
        self.sleep_all_activity_quantile = sleep_all_activity_quantile
        self.sleep_positive_activity_quantile = sleep_positive_activity_quantile
        self.do_sleep_low_temperature_filter = do_sleep_low_temperature_filter
        self.do_short_sleep_filter = do_short_sleep_filter
        self.maximum_offwrist_sleep_proportion = maximum_offwrist_sleep_proportion
        self.ashman_d_minimum = ashman_d_minimum
        self.ashman_d_maximum = ashman_d_maximum
        self.do_temperature_difference_filter = do_temperature_difference_filter
        self.lumus_file_minimum_activity_median_high_proportion = lumus_file_minimum_activity_median_high_proportion
        self.do_description_report_based_filter = do_description_report_based_filter
        self.do_report_border_concentration_filter = do_report_border_concentration_filter
        self.do_report_zero_activity_proportion_filter = do_report_zero_activity_proportion_filter
        self.report_zero_activity_proportion_minimum = report_zero_activity_proportion_minimum
        self.do_report_activity_around_filter = do_report_activity_around_filter
        self.do_report_border_activity_filter = do_report_border_activity_filter
        self.border_concentratition_minimum = border_concentratition_minimum
        self.report_activity_around_minimum = report_activity_around_minimum
        self.do_statistical_measure_filter = do_statistical_measure_filter
        
        self.do_analyze_sleep_borders = do_analyze_sleep_borders

        self.do_valley_peak_algorithm = do_valley_peak_algorithm
        self.valley_quantile = valley_quantile ## update docstring
        self.peak_quantile = peak_quantile ## update docstring

        self.temperature_threshold_refinement_intensity = temperature_threshold_refinement_intensity
        self.temperature_threshold_refinement_quantile = temperature_threshold_refinement_quantile
        self.do_temperature_threshold_refinement = do_temperature_threshold_refinement

        self.do_report_low_activity_proportion_filter = do_report_low_activity_proportion_filter
        self.report_low_activity_proportion_minimum = report_low_activity_proportion_minimum
        self.do_report_low_activity_after = do_report_low_activity_after

        self.do_forbidden_zone = do_forbidden_zone
        self.possible_window_hours = possible_window_hours

        self.do_valley_peak_filter = do_valley_peak_filter

        self.bimodal_maximum_offwrist_proportion = bimodal_maximum_offwrist_proportion
        self.bimodal_minimum_low_activity_proportion = bimodal_minimum_low_activity_proportion
        self.do_temperature_criteria = do_temperature_criteria

        self.do_short_low_temperature_filter = do_short_low_temperature_filter
        self.sleep_low_temperature_proportion_maximum = sleep_low_temperature_proportion_maximum

        self.offwrist_maximum_temperature_difference_median = offwrist_maximum_temperature_difference_median
        self.offwrist_minimum_temperature_difference_median = offwrist_minimum_temperature_difference_median
        self.offwrist_minimum_capsensor1_median = offwrist_minimum_capsensor1_median
        self.offwrist_minimum_capsensor2_median = offwrist_minimum_capsensor2_median
        self.offwrist_maximum_capsensor1_std = offwrist_maximum_capsensor1_std
        self.offwrist_maximum_capsensor2_std = offwrist_maximum_capsensor2_std

        self.half_day_length_validation = half_day_length_validation
        self.decrease_ratio_minimum = decrease_ratio_minimum
        self.next_possible_length_criteria = next_possible_length_criteria
        self.short_valley_peak_offwrist_criteria = short_valley_peak_offwrist_criteria
        self.medium_valley_peak_offwrist_criteria = medium_valley_peak_offwrist_criteria
        self.valley_peak_low_temperature_minimum = valley_peak_low_temperature_minimum
        self.short_offwrist_minimum_decrease_ratio = short_offwrist_minimum_decrease_ratio
        self.lumus_short_offwrist_maximum_capsensor_pvalue = lumus_short_offwrist_maximum_capsensor_pvalue
        self.trust_short_offwrist_maximum_temperature_difference_pvalue = trust_short_offwrist_maximum_temperature_difference_pvalue
        self.short_offwrist_length = short_offwrist_length
        self.offwrists_need_decreasing_temperature = offwrists_need_decreasing_temperature
        self.skip_description_filters = skip_description_filters

        self.input_parameters = self.__dict__.copy()


    def periods_to_df(self,periods):
        """Creates a DataFrame with columns "start" and "end" representing 
        the location of off or onwrist periods present in the input data 
        from a list or np.array.


        Parameters
        ----------
        periods : list or np.array
                In the form [[start0,end0],...,[startN,endN]].

        Returns
        -------
        periods_df : pd.DataFrame
                The same information contained in the input in pd.DataFrame
                formart.
        """

        return pd.DataFrame(periods,columns=["start","end"])

    def df_to_periods(self,periods_df):
        """Creates a np.array containing off or onwrist periods present in 
        the input data  location from a pd.DataFrame.

        Parameters
        ----------
        periods_df : pd.DataFrame
                With at least 2 columns: "start" and "end", for each period.

        Returns
        -------
        periods : np.array
                Information from the "start" and "end" columns.
        """

        return periods_df.loc[:,["start","end"]].values

    def add_datetime_stamps(self,periods_df,mask=None):
        """Adds datetime.datetime information to a DataFrame containing the
        location of off or onwrist periods.


        Parameters
        ----------
        periods_df : pd.DataFrame
                With at least 2 columns: "start" and "end", for each period.

        Returns
        -------
        periods_df_with_datetime : pd.DataFrame
                Inputted DataFrame with added datetime columns.
        """

        periods_df_with_datetime = periods_df.copy()
        ends = periods_df["end"].values

        if mask is None:
                if len(periods_df) > 0:
                        last_index = len(ends)-1
                        if ends[last_index] == self.data_length:
                                ends[last_index] = self.data_length-1

                periods_df_with_datetime.insert(0, "datetime_end", [self.datetime_stamps[o] for o in ends])
                periods_df_with_datetime.insert(0, "datetime_start", [self.datetime_stamps[o] for o in periods_df["start"]])
        else:
                masked_stamps = self.datetime_stamps[mask]
                
                length = len(masked_stamps)

                if len(periods_df) > 0:
                        last_index = len(ends)-1
                        if ends[last_index] == length:
                                ends[last_index] = length-1

                periods_df_with_datetime.insert(0, "datetime_end", [masked_stamps[o] for o in ends])
                periods_df_with_datetime.insert(0, "datetime_start", [masked_stamps[o] for o in periods_df["start"]])

        return periods_df_with_datetime

    def describe_onwrist_periods(self):
        """Computes onwrist period's features that will be used for filtering.

        Returns
        -------
        onwrist_periods_df : pd.DataFrame
                Contains onwrist period's location, epoch length, low activity
                proportion and zero activity proportion.
        """

        onwrist_periods_df = self.periods_to_df(self.onwrist_periods)
        onwrist_periods_df["length"] = onwrist_periods_df["end"]-onwrist_periods_df["start"]
        onwrist_periods_df["low_activity_proportion"] = np.array([below_prop(self.activity[o[0]:o[1]],self.activity_threshold) for o in self.onwrist_periods])
        onwrist_periods_df["zero_activity_proportion"] = np.array([zero_prop(self.activity[o[0]:o[1]]) for o in self.onwrist_periods])

        return onwrist_periods_df

    def print_periods(self,periods):
        """Prints off or onwrist period's location array and, if available,
        also prints the DataFrame version with added datetime stamps.


        Parameters
        ----------
        periods : list or np.array
                In the form [[start0,end0],...,[startN,endN]].
        """

        if self.verbose > 1:
            print(periods)
            if self.datetime_stamps_available and (len(periods) > 0):
                print(self.add_datetime_stamps(self.periods_to_df(periods)))


    def first_stage_refinement(self,):
        """Applies a sequence of filters to the initial onwrist periods pre-
        sent in the input data. Ideally, valid onwrist periods must have a
        minimum length and a low zero activity proportion.
        """
        
        if self.do_onwrist_length_filter:
            self.onwrist_periods_df = self.onwrist_periods_df[self.onwrist_periods_df["length"] > self.minimum_onwrist_length]
        
        if self.do_onwrist_zero_activity_proportion_filter:
            self.onwrist_periods_df = self.onwrist_periods_df[self.onwrist_periods_df["zero_activity_proportion"] < 0.5]
        
        # Reindexing onwrist_periods_df to account for the periods that were filtered out
        self.onwrist_periods_df.index = np.arange(self.onwrist_periods_df.shape[0])
        self.onwrist_periods = self.df_to_periods(self.onwrist_periods_df)

        # Computing refined offwrist periods
        self.first_stage_offwrist = np.zeros(self.data_length)
        for idx in self.onwrist_periods_df.index:
            self.first_stage_offwrist[self.onwrist_periods_df.at[idx,"start"]:self.onwrist_periods_df.at[idx,"end"]] = 1.0
        self.offwrist_periods = zero_sequences(self.first_stage_offwrist)


    def check_valid_border_features(self,i):
        """Checks if, according to input configuration a given epoch has fea-
        tures to classify as a valid border candidate for an offwrist period.

        Parameters
        ----------
        i : int
                Border index
        Returns
        -------
        valid_border_features : boolean
                If True, given index has valid features
        """
        
        valid_border_features = False
        if self.search_border_configuration == "disc":
            if self.activity_median_low[i]:
                valid_border_features = True
        elif self.search_border_configuration == "mod":
            if self.activity_median_low[i] or ((self.temperature[i] < self.temperature_threshold) and (self.activity_median[i] < 2*self.low_activity_threshold)):
                valid_border_features = True
        elif self.search_border_configuration == "mod2":
            if self.activity_median_low[i] or ((self.temperature[i] < self.temperature_threshold) and (self.activity_median[i] < 2*self.low_activity_threshold)) or (self.activity_median[i] == 0):
                valid_border_features = True

        return valid_border_features


    def compute_low_temperature_variance_proportion_around_start(self,start):
        """Computes the proportion of epochs around a window containing the
        location of the offwrist start candidate that have a low temperature
        variance.


        Parameters
        ----------
        start : int
                Start candidate
        Returns
        -------
        low_temperature_variance_proportion_around_start : float
                Computed low temperature variance
        """
        
        if self.centered_low_temperature_variance_proportion:
            low_temperature_variance_proportion_around_start = below_prop(self.normalized_temperature_variance[start-self.half_filter_half_window_length:start+1+self.half_filter_half_window_length],self.temperature_variance_threshold)

        else:
            low_temperature_variance_proportion_around_start = below_prop(self.normalized_temperature_variance[start-self.filter_half_window_length:start+1],self.temperature_variance_threshold)

        if self.verbose > 1:
            print("low_temperature_variance_proportion_around_start",low_temperature_variance_proportion_around_start)
            if start >= 2*self.filter_half_window_length:
                print("low_temperature_variance_proportion_around_start2",below_prop(self.normalized_temperature_variance[start-2*self.filter_half_window_length:start+1],self.temperature_variance_threshold))
            else:
                print("low_temperature_variance_proportion_around_start2",below_prop(self.normalized_temperature_variance[0:start+1],self.temperature_variance_threshold))

        return low_temperature_variance_proportion_around_start


    def compute_low_temperature_variance_proportion_around_end(self,end):
        """Computes the proportion of epochs around a window containing the
        location of the offwrist end candidate that have a low temperature
        variance.


        Parameters
        ----------
        end : int
                End candidate
        Returns
        -------
        low_temperature_variance_proportion_around_end : float
                Computed low temperature variance
        """
        
        if self.centered_low_temperature_variance_proportion:
            low_temperature_variance_proportion_around_end = below_prop(self.normalized_temperature_variance[end-self.half_filter_half_window_length:end+self.half_filter_half_window_length],self.temperature_variance_threshold)
        else:
            low_temperature_variance_proportion_around_end = below_prop(self.normalized_temperature_variance[end:end+self.filter_half_window_length],self.temperature_variance_threshold)

        if self.verbose > 1:
            print("low_temperature_variance_proportion_around_end",low_temperature_variance_proportion_around_end)
            if end + 2*self.filter_half_window_length <= self.data_length:
                print("low_temperature_variance_proportion_around_end2",below_prop(self.normalized_temperature_variance[end:end+2*self.filter_half_window_length],self.temperature_variance_threshold))
            else:
                print("low_temperature_variance_proportion_around_end2",below_prop(self.normalized_temperature_variance[end:self.data_length],self.temperature_variance_threshold))

        return low_temperature_variance_proportion_around_end


    def start_peak_found(self,start,previous_end):
        """The sequence of steps for when an "ideal" start candidate is found.
        The initial candidate has a high temperature variance and it's in a re-
        gion that shares this feature, indicating the presence of a local maxi-
        ma. The refined candidate is the most probable location of the maxima.

        Parameters
        ----------
        start : int
                Initial candidate.
        previous_end : int
                Location of the ending of the preceding offwrist period.

        Returns
        -------
        search_start : boolean
                If True, the algorithm keeps trying to refine the start of an
                offwrist period
        """
        
        start = start - self.filter_half_window_length + np.argmax(self.normalized_temperature_variance[start-self.filter_half_window_length:start+1])
        if start < previous_end:
            start = previous_end+2

        self.refined_offwrist_periods.append([start,0])
        self.print_periods(self.refined_offwrist_periods)

        search_start = False

        return search_start


    def end_peak_found(self,end,offwrist_index):
        """The sequence of steps for when an "ideal" end candidate is found.
        The initial candidate has a high temperature variance and it's in a re-
        gion that shares this feature, indicating the presence of a local maxi-
        ma. The refined candidate is the most probable location of the maxima.

        Parameters
        ----------
        end : int
                Initial candidate.
        offwrist_index : int
                Index of the offwrist period to be refined

        Returns
        -------
        search_start : boolean
                If True, the algorithm will try to refine the start of an
                offwrist period
        """
        
        end = end + np.argmax(self.normalized_temperature_variance[end:end+self.filter_half_window_length+1])
        self.refined_offwrist_periods[-1][1] = end
        self.print_periods(self.refined_offwrist_periods)
        
        search_start = True
        offwrist_index += 1

        return search_start,offwrist_index


    def check_offwrist_too_close(self,following_onwrist_period_length,end,next_start):
        """Checks if an offwrist period is too close to the following one. If
        it is, they will be joined together. An extra low activity criteria may
        be applied.


        Parameters
        ----------
        following_onwrist_period_length : int
                Distance in epochs between offwrist periods
        end : int
                Location of the end candidate for the current offwrist period
        next_start : int
                Location of start candidate for the next offwrist period to be
                refined.
        
        Returns
        -------
        too_close_offwrist : boolean
                If True, current offwrist period and following one will be com-
                bined.
        """
        
        too_close_offwrist = False
        if self.too_close_offwrist_low_activity_criteria:
            if (following_onwrist_period_length <= self.minimum_preceding_onwrist_period_length) and (below_prop(self.activity[end:next_start],self.activity_threshold) > self.minimum_low_activity_proportion):
                too_close_offwrist = True
        else:
            if (following_onwrist_period_length <= self.minimum_preceding_onwrist_period_length):
                too_close_offwrist = True

        return too_close_offwrist

    def do_more_checks_around_end(self,offwrist_index,next_start,end):
        """Sequence of steps for when a peak can't be determined in the epochs
        around or preceding the end candidate: check if the following onwrist
        period is valid and if it is, tries to find a peak towards it.

        Parameters
        ----------
        offwrist_index : int
                Index of the offwrist period to be refined
        next_start : int
                Location of start candidate for the next offwrist period to be
                refined.
        end : int
                Location of the end candidate for the current offwrist period
        
        Returns
        -------
        search_start : boolean
                If True, the algorithm keeps trying to refine the start of an
                offwrist period
        offwrist_index : int
                Index of the offwrist period to be refined
        """
        
        search_start = False

        following_onwrist_period_length = next_start - end

        too_close_offwrist = self.check_offwrist_too_close(following_onwrist_period_length,end,next_start)

        if too_close_offwrist:
            # When refining the end of an offwrist period, deleting its index
            # from the list has the effect of combining it to the following one
            # because in the refine offwrist periods list, a start has been de-
            # fined, but the end candidate will now be the one from the next 
            # offwrist period.
            self.offwrist_periods = np.delete(self.offwrist_periods,offwrist_index,axis=0)
            if self.verbose > 1:
                print("delete")
        else:
            # If the offwrist periods have a valid distance between then, the
            # epochs preceding the candidate are evaluated for their tempera-
            # ture variance. This computation is in the opposite direction in
            # relation to the compute_low_temperature_variance_proportion_a-
            # round_end module.
            low_temperature_variance_proportion_around_end = below_prop(self.normalized_temperature_variance[end-self.filter_half_window_length-1:end],self.temperature_variance_threshold)
            if self.verbose > 1:
                print("low_temperature_variance_proportion_around_end",low_temperature_variance_proportion_around_end)
            if low_temperature_variance_proportion_around_end > self.maximum_low_temperature_variance_proportion_around_border:
                # If the temperature variance doesn't meet the criteria, the
                # offwrist is combined to the next one.
                self.offwrist_periods = np.delete(self.offwrist_periods,offwrist_index,axis=0)
                if self.verbose > 1:
                    print("delete")
            else:
                # If the criteria is met, the end candidate is defined.
                self.refined_offwrist_periods[-1][1] = end-self.filter_half_window_length-1 + np.argmax(self.normalized_temperature_variance[end-self.filter_half_window_length-1:end])
                self.print_periods(self.refined_offwrist_periods)

                search_start = True
                offwrist_index += 1

        return offwrist_index,search_start


    def find_peak_base(self,start,offwrist_index,previous_end):
        """Sequence of steps for when a peak can't be determined in the epochs
        around or following the start candidate: search for a valid candidate
        in the epochs before the initial candidate.

        Parameters
        ----------
        start : int
                Location of the start candidate for the current offwrist period
        offwrist_index : int
                Index of the offwrist period to be refined
        previous_end : int
                Location of the ending of the preceding offwrist period.

        Returns
        -------
        search_start : boolean
                If True, the algorithm keeps trying to refine the start of an
                offwrist period
        """
        
        search_start = True

        i = start-1
        last_valid_border_index = start
        peak_base = start

        # The search is limited by the end of the previously refined offwrist
        # period.
        while i > previous_end:
            # In order for an epoch to be a candidate for offwrist period bor-
            # der, it must meet some criteria related to its activity and tem-
            # perature characteristics.
            valid_border_features = self.check_valid_border_features(i)

            if valid_border_features:
                # In addition to that, it must also meet the basic criteria for
                # offwrist borders: high temperature variance. The first valid
                # epoch that meets all criteria is defined as peak base and the
                # search is halted with a succesful result.
                if self.normalized_temperature_variance[i] >= self.temperature_variance_threshold:
                    peak_base = i
                    i = previous_end
                i -= 1

            else:
                # This condition is met when an invalid border epoch is reached
                # after a sequence of valid ones: the search has failed.
                last_valid_border_index = i
                i = previous_end

        if last_valid_border_index == start:
            if peak_base < start:
                # If the search is succesful, a new search follows to determine
                # the location of the peak from the base.
                if self.verbose > 1:
                    print("peak found")
                peak = peak_base-1
                while self.normalized_temperature_variance[peak] >= self.normalized_temperature_variance[peak+1]:
                    peak -= 1

                self.refined_offwrist_periods.append([peak,0])
                self.print_periods(self.refined_offwrist_periods)

                search_start = False
            else:
                # If the search finds valid border features but is not capable
                # of finding valid temperature variance features, the offwrist
                # period is filtered out.
                self.offwrist_periods = np.delete(self.offwrist_periods,offwrist_index,axis=0)
                if self.verbose > 1:
                    print("delete")

        else:
            # If the search fails to find a peak base, the best possible candi-
            # date is defined as the start.
            if self.verbose > 1:
                print("peak not found, no activity_median_low")

            self.refined_offwrist_periods.append([last_valid_border_index,0]) 
            self.print_periods(self.refined_offwrist_periods)

            search_start = False


        return search_start


    def try_find_peak_base(self,offwrist_index,next_start,end):
        """Sequence of steps for when an initial end cadidate doesn't meet basic
        criteria: check if the following onwrist period is valid and if it is, 
        tries to find a peak towards it.

        Parameters
        ----------
        offwrist_index : int
                Index of the offwrist period to be refined
        next_start : int
                Location of start candidate for the next offwrist period to be
                refined.
        end : int
                Location of the end candidate for the current offwrist period

        Returns
        -------
        search_start : boolean
                If True, the algorithm keeps trying to refine the start of an
                offwrist period
        offwrist_index : int
                Index of the offwrist period to be refined
        """
        
        search_start = False

        following_onwrist_period_length = next_start - end
        
        too_close_offwrist = self.check_offwrist_too_close(following_onwrist_period_length,end,next_start)

        if too_close_offwrist:
            # When refining the end of an offwrist period, deleting its index
            # from the list has the effect of combining it to the following one
            # because in the refine offwrist periods list, a start has been de-
            # fined, but the end candidate will now be the one from the next 
            # offwrist period.
            self.offwrist_periods = np.delete(self.offwrist_periods,offwrist_index,axis=0)
            if self.verbose > 1:
                print("delete")
        else:
            if self.verbose > 1:
                print("searching peak")

            i = end+1
            last_valid_border_index = end
            peak_base = end
            while i < next_start:
                # In order for an epoch to be a candidate for offwrist period 
                # border, it must meet some criteria related to its activity 
                # and temperature characteristics.
                valid_border_features = self.check_valid_border_features(i)

                if valid_border_features:
                    # In addition to that, it must also meet the basic crite-
                    # ria for offwrist borders: high temperature variance. The 
                    # first valid epoch that meets all criteria is defined as 
                    # peak base and the search is halted with a succesful re-
                    # sult.
                    if self.normalized_temperature_variance[i] >= self.temperature_variance_threshold:
                        peak_base = i
                        i = next_start
                    i += 1
                     
                else:
                    # This condition is met when an invalid border epoch is 
                    # reached after a sequence of valid ones: the search has 
                    # failed.
                    last_valid_border_index = i
                    i = next_start

            if last_valid_border_index == end:
                if peak_base > end:
                    # If the search is succesful, a new search follows to de-
                    # termine the location of the peak from the base.
                    if self.verbose > 1:
                        print("peak found")
                    peak = peak_base+1
                    while self.normalized_temperature_variance[peak] >= self.normalized_temperature_variance[peak-1]:
                        peak += 1

                    self.refined_offwrist_periods[-1][1] = peak
                    self.print_periods(self.refined_offwrist_periods)

                    search_start = True
                    offwrist_index += 1
                else:
                    # If the search finds valid border features but is not ca-
                    # pable of finding valid temperature variance features, the 
                    # offwrist period is filtered out.
                    self.offwrist_periods = np.delete(self.offwrist_periods,offwrist_index,axis=0)
                    if self.verbose > 1:
                        print("delete")

            else:
                # If the search fails to find a peak base, the best possible can-
                # didate is defined as the start.
                if self.verbose > 1:
                    print("peak not found, no activity_median_low")

                self.refined_offwrist_periods[-1][1] = last_valid_border_index
                self.print_periods(self.refined_offwrist_periods)

                search_start = True
                offwrist_index += 1

        return offwrist_index,search_start


    def search_offwrist_start(self,offwrist_index):
        """Tries to find, in the neighborhood of a given an initial candidate,
        the best epoch to define as the start of an offwrist period. If there
        aren't any valid candidates closeby, the offwrist period may be fil-
        tered out. The ideal start candidate is an epoch with a large tempe-
        rature variance indicating the typical drop in temperature we see in
        the transition from onwrist to offwrist.

        Parameters
        ----------
        offwrist_index : int
                Index of the offwrist period to be refined
        
        Returns
        -------
        search_start : boolean
                If True, the algorithm keeps trying to refine the start of an
                offwrist period
        offwrist_index : int
                Index of the offwrist period to be refined
        """
        
        # If this succesfully finds the best start candidate, this boolean will
        # switch value.
        search_start = True

        # Offwrist periods are in the form [[start0,end0],...,[startN,endN]]
        start = self.offwrist_periods[offwrist_index][0]

        # The search for the start candidate can't get to the end of the pre-
        # vious offwrist period.
        if len(self.refined_offwrist_periods) > 0:
            previous_end = self.refined_offwrist_periods[-1][1]
            if start < previous_end:
                start = previous_end+2
        else:
            previous_end = 0

        if self.verbose > 1:
            print("start",start,self.normalized_temperature_variance[start])

        # First, check if the offwrist period is too close to the beggining
        # of the input data.
        if start >= self.filter_half_window_length:
            # Second, check if the temperature variance is high.
            if self.normalized_temperature_variance[start] >= self.temperature_variance_threshold:
                # If the candidate has a high temperature variance, check the 
                # neighoring (around or previous) epochs, we're interested in
                # finding the location of a temperature variance peak.
                low_temperature_variance_proportion_around_start = self.compute_low_temperature_variance_proportion_around_start(start)
                if (low_temperature_variance_proportion_around_start <= self.maximum_low_temperature_variance_proportion_around_border):
                    # If the neighborhood has a low proportion of low tempera-
                    # ture variance values, it's a peak.
                    search_start = self.start_peak_found(start,previous_end)
                else:
                    # If the neighborhood doesn't have a low proportion of low 
                    # temperature variance values, either the candidate is in-
                    # valid and filtered out or we look for a closeby peak.
                    if self.quick_delete_start:
                        self.offwrist_periods = np.delete(self.offwrist_periods,offwrist_index,axis=0)
                        if self.verbose > 1:
                            print("delete")
                    else:
                        if self.verbose > 1:
                            print("searching peak")
                        search_start = self.find_peak_base(start,offwrist_index,previous_end)

            else:
                # If the temperature variance is low, we try and find a closeby
                # peak.
                if self.verbose > 1:
                    print("searching peak")

                if offwrist_index > 0:
                    previous_end = self.offwrist_periods[offwrist_index-1][1]
                else:
                    previous_end = 0

                search_start = self.find_peak_base(start,offwrist_index,previous_end)
                
        else:
            if self.verbose > 1:
                print("low_temperature_variance_proportion_around_start",below_prop(self.normalized_temperature_variance[0:start+1],self.temperature_variance_threshold))

            # If it is too close, the start candidate will be the epoch with
            # the largest temperature variance.
            start = np.argmax(self.normalized_temperature_variance[0:start+1])
            self.refined_offwrist_periods.append([start,0])
            self.print_periods(self.refined_offwrist_periods)

            search_start = False

        return search_start,offwrist_index

    def search_offwrist_end(self,offwrist_index):
        """"Tries to find, in the neighborhood of a given an initial candidate,
        the best epoch to define as the end of an offwrist period. If there
        aren't any valid candidates closeby, the offwrist period may be fil-
        tered out. The ideal end candidate is an epoch with a large temperatu-
        re variance indicating the typical rise in temperature we see in the 
        transition from offwrist to onwrist.

        Parameters
        ----------
        offwrist_index : int
                Index of the offwrist period to be refined
        
        Returns
        -------
        search_start : boolean
                If True, the algorithm keeps trying to refine the start of an
                offwrist period.
        offwrist_index : int
                Index of the offwrist period to be refined.
        """
        
        # If this succesfully finds the best end candidate, this boolean will
        # switch value.
        search_start = False

        offwrist_count = len(self.offwrist_periods)

        # Offwrist periods are in the form [[start0,end0],...,[startN,endN]]
        end = self.offwrist_periods[offwrist_index][1]

        if self.verbose > 1:
            print("end", end, self.normalized_temperature_variance[end])

        # First, check if the temperature variance is high.
        if self.normalized_temperature_variance[end] >= self.temperature_variance_threshold:
            if self.verbose > 1:
                print("above thresh")
            # Second, check if the offwrist period is too close to the ending
            # of the input data.
            if end + self.filter_half_window_length <= self.data_length:
                # If the candidate has a high temperature variance, check the 
                # neighoring (around or following) epochs, we're interested in 
                # finding the location of a temperature variance peak.
                low_temperature_variance_proportion_around_end = self.compute_low_temperature_variance_proportion_around_end(end)
                if low_temperature_variance_proportion_around_end <= self.maximum_low_temperature_variance_proportion_around_border:
                    # If the neighborhood has a low proportion of low tempera-
                    # ture variance values, it's a peak.
                    search_start,offwrist_index = self.end_peak_found(end,offwrist_index)
                else:
                    # If the neighborhood doesn't have a low proportion of low 
                    # temperature variance values, either the candidate is in-
                    # valid and filtered out or we look for a closeby peak.
                    if offwrist_index < self.offwrist_periods.shape[0] - 1:
                        # If the current offwrist period isn't the last one, we
                        # will check if it's too close to the following period 
                        # and if it's not, we'll search for a candidate in the 
                        # previous epochs.
                        next_start = self.offwrist_periods[offwrist_index + 1][0]
                        offwrist_index,search_start = self.do_more_checks_around_end(offwrist_index,next_start,end)
                    else:
                        # If this is the last period, we'll choose the epoch with
                        # the highest variance between the initial candidate and
                        # the end of the input data
                        end = end + np.argmax(self.normalized_temperature_variance[end:self.data_length])
                        self.refined_offwrist_periods[-1][1] = end
                        self.print_periods(self.refined_offwrist_periods)

                        search_start = True
                        offwrist_index += 1
            else:
                # If it is too close, the end candidate will be the epoch with
                # the highest temperature variance.
                if self.verbose > 1:
                    print("low_temperature_variance_proportion_around_end",below_prop(self.normalized_temperature_variance[end:self.data_length],self.temperature_variance_threshold))

                end = end + np.argmax(self.normalized_temperature_variance[end:self.data_length])
                self.refined_offwrist_periods[-1][1] = end
                self.print_periods(self.refined_offwrist_periods)

                search_start = True
                offwrist_index += 1
        else:
            if self.verbose > 1:
                print("below thresh")

            if self.verbose > 1:
                if end + self.filter_half_window_length <= self.data_length:
                    print("low_temperature_variance_proportion_around_end",below_prop(self.normalized_temperature_variance[end:end + self.filter_half_window_length],self.temperature_variance_threshold))
                    if end + 2*self.filter_half_window_length <= self.data_length:
                        print("low_temperature_variance_proportion_around_end2",below_prop(self.normalized_temperature_variance[end:end + 2*self.filter_half_window_length],self.temperature_variance_threshold))
                    else:
                        print("low_temperature_variance_proportion_around_end2",below_prop(self.normalized_temperature_variance[end:self.data_length],self.temperature_variance_threshold))
                else:
                    print("low_temperature_variance_proportion_around_end",below_prop(self.normalized_temperature_variance[end:self.data_length],self.temperature_variance_threshold))

            # If the temperature variance is low and the current offwrist period 
            # isn't the last one, we will search for a peak in the neighborhood.
            if offwrist_index < self.offwrist_periods.shape[0] - 1:
                next_start = self.offwrist_periods[offwrist_index + 1][0]
                offwrist_index,search_start = self.try_find_peak_base(offwrist_index,next_start,end)

            else:
                # If it's the last one, we'll choose the epoch with the highest 
                # temperature variance.
                end = end + np.argmax(self.normalized_temperature_variance[end:self.data_length])
                self.refined_offwrist_periods[-1][1] = end
                self.print_periods(self.refined_offwrist_periods)

                search_start = True
                offwrist_index += 1

        return search_start,offwrist_index


    def second_stage_refinement(self,):
        """Sequentially refines the border of all the offwrist periods present
        in the input data. Offwrists with invalid initial borders may be fil-
        tered out. Also, offwrist periods that are too close may be combined to
        one;
        """
        
        # Border candidates may be evaluated with a centered window that will
        # extend to half of the window lenght used in the filters that previous-
        # ly extracted features from the data.
        self.half_filter_half_window_length = int(round(self.filter_half_window_length/2))

        offwrist_index = 0
        search_start = True
        while offwrist_index < self.offwrist_periods.shape[0]:
            if self.verbose > 1:
                print(offwrist_index,self.offwrist_periods[offwrist_index])
            
            if search_start:
                search_start,offwrist_index = self.search_offwrist_start(offwrist_index)

            else:
                search_start,offwrist_index = self.search_offwrist_end(offwrist_index)

    def refine_temperature_threshold(self,):
        """Applies a sequence of filters to the initial onwrist periods pre-
        sent in the input data. Ideally, valid onwrist periods must have a
        minimum length and a low zero activity proportion.
        """

        self.refined_offwrist_periods = np.array(self.refined_offwrist_periods)
        self.refined_offwrist_periods_df = self.periods_to_df(self.refined_offwrist_periods)
        self.refined_offwrist_periods_df["length"] = self.refined_offwrist_periods_df["end"] - self.refined_offwrist_periods_df["start"]

        if self.verbose:
                print("self.refined_offwrist_periods_df")
                print(self.refined_offwrist_periods_df)

        # Second stage refined offwrist periods will be grouped according to
        # length thresholding. Long offwrists statistical properties tend to
        # be more accurately estimated.
        self.refined_long_offwrist_periods_df = self.refined_offwrist_periods_df[self.refined_offwrist_periods_df["length"] >= self.long_offwrist_length]

        if self.verbose:
                print("self.refined_long_offwrist_periods_df")
                print(self.refined_long_offwrist_periods_df)


        self.long_offwrist_temperature = 0
        refined_temperature_threshold = self.temperature_threshold
        if len(self.refined_long_offwrist_periods_df) > 0:
                self.refined_long_offwrist_periods = self.df_to_periods(self.refined_long_offwrist_periods_df)

                self.refined_offwrist = np.ones(self.data_length)
                for offwrist_index in self.refined_offwrist_periods:
                    self.refined_offwrist[offwrist_index[0]:offwrist_index[1]] = 0.0


                self.refined_long_offwrist = np.ones(self.data_length)
                for offwrist_index in self.refined_long_offwrist_periods:
                    self.refined_long_offwrist[offwrist_index[0]:offwrist_index[1]] = 0.0

                refined_long_offwrist_bool = np.logical_not(self.refined_long_offwrist.astype(bool))

                # Using only the temperature information from long offwrist,
                # a high quantile will be selected as the threshold of high
                # temperature values for an offwrist period.
                self.long_offwrist_temperature = np.quantile(self.temperature[refined_long_offwrist_bool],self.temperature_threshold_refinement_quantile,method="linear")

                # The threshold will be refined by bridging the gap between
                # the initial computed threshold and the estimator of the 
                # offwrist temperature level.
                temperature_threshold_gap = 0
                if self.temperature_threshold > self.long_offwrist_temperature:
                        temperature_threshold_gap = self.temperature_threshold-self.long_offwrist_temperature
                        refined_temperature_threshold = self.long_offwrist_temperature + self.temperature_threshold_refinement_intensity*temperature_threshold_gap
                # else:

        if self.do_temperature_threshold_refinement:
                self.temperature_threshold = refined_temperature_threshold

    def compute_sleep_low_activity_threshold(self):
        """Computes the sleep low activity threshold according to given input
        configuration. The threshold is calculated based on a specific quantile
        of the activty data. The quantile is chosen based on user inputted pa-
        rameters.


        Returns
        -------
        sleep_low_activity_threshold : float
                Level of activity that's considered low enough to indicate that
                the subject is asleep.
        """
        
        if self.sleep_low_activity_threshold_configuration == "all":
            sleep_low_activity_threshold = np.quantile(self.activity,self.sleep_all_activity_quantile,method="linear")
        else:
            if self.sleep_low_activity_threshold_configuration == "zp":
                activity_threshold_quantile = self.activity_zero_proportion + (1.0 - self.activity_zero_proportion)*self.sleep_positive_activity_quantile
                sleep_low_activity_threshold = np.quantile(self.activity,activity_threshold_quantile,method="linear")
            elif self.sleep_low_activity_threshold_configuration == "both":
                if self.activity_zero_proportion < self.sleep_all_activity_quantile:
                    sleep_low_activity_threshold = np.quantile(self.activity,self.sleep_all_activity_quantile,method="linear")
                else:
                    activity_threshold_quantile = self.activity_zero_proportion + (1.0 - self.activity_zero_proportion)*self.sleep_positive_activity_quantile
                    sleep_low_activity_threshold = np.quantile(self.activity,activity_threshold_quantile,method="linear")

        return sleep_low_activity_threshold


    def sleep_low_temperature_filter(self):
        """Sleep periods, like offwrist periods, have a lower mean tempera-
        ture level. Sleep period borders will then be refined to ensure that
        they reflect this.
        """
        
        if self.do_sleep_low_temperature_filter:
            sleep_periods = zero_sequences(self.estimated_sleep)

            for s in sleep_periods:
                start = s[0]
                end = s[1]

                while (start < self.data_length) and (self.temperature[start] < self.temperature_threshold):
                    self.estimated_sleep[start] = 1.0
                    start += 1

                while (end < self.data_length) and (self.temperature[end] < self.temperature_threshold):
                    self.estimated_sleep[end] = 1.0
                    end -= 1


    def estimate_sleep_then_filter(self):
        """Estimated sleep periods are (initially) regions on the valid in-
        put data that mantain a low activity level for 4 hours or more. Off-
        wrist periods that highly overlap sleep periods will be filtered out. 
        """
        
        # Valid activity is padded with the largest acitivity value present
        # in the input data to bias the borders towards wake, then the pad-
        # ded signal goes through a median filter.
        padding = np.max(self.activity)*np.ones(2*self.sleep_activity_filter_half_window_length)
        padded_valid_activity = np.insert(self.valid_activity,0,padding)
        padded_valid_activity = np.append(padded_valid_activity,padding)
        filtered_valid_activity = median_filter(padded_valid_activity,self.sleep_activity_filter_half_window_length,padding='padded')

        # The threshold for low activity that is related to sleep is comp-
        # uted using quantile measurements
        sleep_low_activity_threshold = self.compute_sleep_low_activity_threshold()
        self.sleep_low_activity_threshold = sleep_low_activity_threshold

        # Filtered valid activity is then thresholded to select the initial
        # estimated sleep periods.
        sleep_estimate = np.where(filtered_valid_activity > sleep_low_activity_threshold, 1, 0)
        sleep_periods = zero_sequences(sleep_estimate)
        sleep_periods_df = self.periods_to_df(sleep_periods)
        sleep_periods_df = self.add_datetime_stamps(sleep_periods_df,self.short_offwrist_only_mask)
        sleep_periods_df["length"] = sleep_periods_df["end"] - sleep_periods_df["start"]

        if self.verbose:
                print("initial sleep_periods")
                print(sleep_periods_df)
        if self.do_short_sleep_filter:
            sleep_periods_df = sleep_periods_df[sleep_periods_df["length"] > self.long_offwrist_length]
            sleep_periods_df.index = np.arange(len(sleep_periods_df))

        if self.verbose:
                print("length filtered sleep_periods")
                print(sleep_periods_df)

        # Proportion of low temperature filtering succeeds.
        sleep_periods = self.df_to_periods(sleep_periods_df)
        sleep_periods_df["low_temperature_proportion"] = np.array([below_prop(self.valid_temperature[s[0]:s[1]],self.temperature_threshold) for s in sleep_periods])        
        if self.do_short_low_temperature_filter:
            sleep_periods_df = sleep_periods_df[sleep_periods_df["low_temperature_proportion"] < self.sleep_low_temperature_proportion_maximum]
            sleep_periods_df.index = np.arange(len(sleep_periods_df))

        if self.verbose:
                print("temperature filtered sleep_periods")
                print(sleep_periods_df)


        # After filtering, improved sleep estimate is obtained.
        sleep_periods = self.df_to_periods(sleep_periods_df)
        sleep_estimate = np.ones(len(sleep_estimate))
        for period in sleep_periods:
            start,end = period
            sleep_estimate[start:end] = 0.0

        self.estimated_sleep = np.ones(self.data_length)
        self.estimated_sleep[self.short_offwrist_only_mask] = sleep_estimate

        # Final sleep estimate is obtained after one last filtering stage.
        self.sleep_low_temperature_filter()

        sleep_periods = zero_sequences(self.estimated_sleep)
        self.sleep_periods = self.periods_to_df(sleep_periods)
        self.sleep_periods = self.add_datetime_stamps(self.sleep_periods)

        if self.verbose:
                print("self.sleep_periods\n",self.sleep_periods)


        # Low activity levels for thresholding are refined according to es-
        # timated sleep epochs.
        refined_low_activity_quantile = 0.1
        self.sleep_activity = self.activity[np.where(self.estimated_sleep > 0,False,True)]
        if len(self.sleep_activity) > 0:
                sleep_activity_zero_proportion = zero_prop(self.sleep_activity)
                activity_threshold_quantile = sleep_activity_zero_proportion + (1.0 - sleep_activity_zero_proportion)*refined_low_activity_quantile
                self.refined_low_activity_threshold = np.quantile(self.sleep_activity,activity_threshold_quantile,method="linear")
        else:
                self.refined_low_activity_threshold = self.activity_threshold


        # Offwrist periods that are contained in estimated sleep periods are
        # probable wrong detections.
        offwrist_period_sleep_proportion = np.array([zero_prop(self.estimated_sleep[offwrist_index[0]:offwrist_index[1]]) for offwrist_index in self.after_initial_refinement_offwrist_periods])
        valid_sleep_proportion_mask = np.where(offwrist_period_sleep_proportion < self.maximum_offwrist_sleep_proportion,True,False)

        # Sleep filtered offwrist periods may be spared if they meet some cri-
        # teria.
        for i in range(len(valid_sleep_proportion_mask)):
            if not valid_sleep_proportion_mask[i]:
                if self.is_lumus_file:
                    # The numerical threshold used here was obtained empirically
                    # using historical data
                    if self.do_capacitive_sensor_variance_filter and self.after_initial_refinement_offwrist_report.at[i,"capacitive_difference_variance"] < 1.75e-4:
                         valid_sleep_proportion_mask[i] = True
                        
        self.sleep_filtered_offwrist_periods = self.after_initial_refinement_offwrist_periods[valid_sleep_proportion_mask].copy()
        sleep_filter_deleted_periods = 1-valid_sleep_proportion_mask.astype(int)
        self.sleep_filter_deleted_periods = sleep_filter_deleted_periods.nonzero()[0]


    def low_temperature_filter(self,):
        """Valid offwrist periods must have a high proportion of low tempera-
        ture values. This is related to the fact that the environment tempera-
        ture is usually significantly lower than human skin temperature.
        """
        
        if self.verbose > 1:
            print("refined offwrist_periods and props\n",self.refined_offwrist_periods)
            if self.datetime_stamps_available:
                print(self.add_datetime_stamps(self.refined_offwrist_periods_df))
            print("offwrist_minimum_low_temperature_proportion",self.offwrist_minimum_low_temperature_proportion)
        
        valid_indexes = []
        for offwrist in self.refined_offwrist_periods_df.index:
                if (self.half_day_length_validation and (self.refined_offwrist_periods_df.at[offwrist,"length"] >= self.half_day_length)):
                    valid_indexes.append(offwrist)
                else:
                    if self.refined_offwrist_periods_df.at[offwrist,"low_temperature_proportion"] >= self.offwrist_minimum_low_temperature_proportion:
                        valid_indexes.append(offwrist)

        self.refined_offwrist_periods_df = self.refined_offwrist_periods_df.loc[valid_indexes,:]
        self.refined_offwrist_periods_df.index = np.arange(self.refined_offwrist_periods_df.shape[0])
        self.refined_offwrist_periods = self.df_to_periods(self.refined_offwrist_periods_df)
        
        if self.verbose > 1:
            print("temperature filtered refined offwrist_periods\n",self.refined_offwrist_periods)
            if self.datetime_stamps_available:
                print(self.add_datetime_stamps(self.refined_offwrist_periods_df))

    def low_activity_filter(self,):
        """Valid offwrist periods must have a high proportion of low activity
        values. An additional criteria may be applied: it must also contain a
        high proportion of zero activity values.
        """
        
        if self.do_low_activity_filter:
                if self.verbose > 1:
                        print("filtered refined offwrist_periods act_bp")
                        print("minimum_low_activity_proportion",self.minimum_low_activity_proportion)
                        print("act_bp filtered refined offwrist_periods",self.refined_offwrist_periods_df["low_activity_proportion"])
                        print("act_zp filtered refined offwrist_periods",self.refined_offwrist_periods_df["zero_activity_proportion"])
                
                if self.datetime_stamps_available:
                        print(self.add_datetime_stamps(self.refined_offwrist_periods_df))

            
                self.refined_offwrist_periods_df = self.refined_offwrist_periods_df[self.refined_offwrist_periods_df["low_activity_proportion"] > self.minimum_low_activity_proportion]
                
                # valid_indexes = []
                # for offwrist in self.refined_offwrist_periods_df.index:
                #         if self.refined_offwrist_periods_df.at[offwrist,"low_activity_proportion"] >= self.minimum_low_activity_proportion:
                #                 valid_indexes.append(offwrist)
                #         else:
                #                 if self.refined_offwrist_periods_df.at[offwrist,"zero_activity_proportion"] >= 0.7:
                #                         valid_indexes.append(offwrist)

                # self.refined_offwrist_periods_df = self.refined_offwrist_periods_df.loc[valid_indexes,:]

                if self.do_zero_activity_filter:
                        self.refined_offwrist_periods_df = self.refined_offwrist_periods_df[self.refined_offwrist_periods_df["zero_activity_proportion"] > 0.7]

                self.refined_offwrist_periods_df.index = np.arange(self.refined_offwrist_periods_df.shape[0])
                self.refined_offwrist_periods = self.df_to_periods(self.refined_offwrist_periods_df)

                if self.verbose > 1:
                        print("act filtered refined offwrist_periods\n",self.refined_offwrist_periods)
                        if self.datetime_stamps_available:
                                print(self.add_datetime_stamps(self.refined_offwrist_periods_df))

    def low_temperature_variance_filter(self,):
        """Valid offwrist periods borders can't have low temperature variance
        values. This is related to the consistent drop in temperature observed
        in the transition from onwrist to offwrist along with the correspon-
        ding rise from offwrist to onwrist.
        """
        
        if self.do_low_temperature_variance_filter:
            low_temperature_variance_threshold = np.quantile(self.normalized_temperature_variance,0.4,method="linear")
            offwrist_border_temperature_variance = np.array([[self.normalized_temperature_variance[o[0]],self.normalized_temperature_variance[o[1]]] for o in self.refined_offwrist_periods])
            is_offwrist_border_temperature_variance_high = np.array([((o[0] > low_temperature_variance_threshold) and (o[1] > low_temperature_variance_threshold)) for o in offwrist_border_temperature_variance])
            if self.verbose > 1:
                print("temperature_variance_threshold",self.temperature_variance_threshold)
                print("low_temperature_variance_threshold",low_temperature_variance_threshold)
                print("offwrist_border_temperature_variance\n",offwrist_border_temperature_variance)
                print("is_offwrist_border_temperature_variance_high",is_offwrist_border_temperature_variance_high)

            self.refined_offwrist_periods_df = self.refined_offwrist_periods_df.loc[is_offwrist_border_temperature_variance_high,:].copy()
            self.refined_offwrist_periods_df.index = np.arange(self.refined_offwrist_periods_df.shape[0])
            self.refined_offwrist_periods = self.df_to_periods(self.refined_offwrist_periods_df)

            if self.verbose > 1:
                print("do_low_temperature_variance_filter offwrist_periods\n",self.refined_offwrist_periods)
                if self.datetime_stamps_available:
                    print(self.add_datetime_stamps(self.periods_to_df(self.refined_offwrist_periods)))

    def offwrist_length_filter(self,):
        """Valid offwrist periods must have a minimum length. Too short off-
        wrist-like periods may appear as a result of hardware failures.
        """
        
        if self.do_offwrist_length_filter:
            self.refined_offwrist_periods_df = self.refined_offwrist_periods_df[self.refined_offwrist_periods_df["length"] >= self.minimum_offwrist_length]
            self.refined_offwrist_periods_df.index = np.arange(self.refined_offwrist_periods_df.shape[0])
            self.refined_offwrist_periods = self.df_to_periods(self.refined_offwrist_periods_df)


    def analyze_sleep_borders(self,half_window_size=2,minimum_sleep_border_offwrist_length=3):
        """Sleep period estimation may be wrong due to the presence of short
        offwrists on the borders of each period. To handle that, special fil-
        ters are applied to try and detect these offwrists.
        """

        window_size = 2*half_window_size + 1

        self.high_temperature_variance_threshold = np.quantile(self.normalized_temperature_variance,0.95,method='inverted_cdf')
        is_temperature_variance_high = np.where(self.normalized_temperature_variance > self.high_temperature_variance_threshold,1,0)        
        is_temperature_variance_high_rolling = rolling_sum(is_temperature_variance_high,half_window_size)
        is_temperature_variance_sustained_high = np.where(is_temperature_variance_high_rolling == window_size,True,False)
        
        if self.verbose:
                print("self.activity",self.activity)
                print("self.activity_median",self.activity_median)
        forward_activity_median = median_filter(self.activity,10,center=False,forward=True)
        if self.verbose:
                print("self.forward_activity_median",forward_activity_median)
        backward_activity_median = median_filter(self.activity,10,center=False,forward=False)
        if self.verbose:
                print("self.backward_activity_median",backward_activity_median)
        # input()

        is_activity_median_zero = np.where(self.activity_median == 0,1,0)        
        is_activity_median_zero_rolling = rolling_sum(is_activity_median_zero,half_window_size)
        is_activity_median_sustained_zero = np.where(is_activity_median_zero_rolling == window_size,True,False)

        is_forward_activity_median_zero = np.where(forward_activity_median == 0,1,0)        
        is_forward_activity_median_zero_rolling = rolling_sum(is_forward_activity_median_zero,half_window_size)
        is_forward_activity_median_sustained_zero = np.where(is_forward_activity_median_zero_rolling == window_size,True,False)

        is_backward_activity_median_zero = np.where(backward_activity_median == 0,1,0)        
        is_backward_activity_median_zero_rolling = rolling_sum(is_backward_activity_median_zero,half_window_size)
        is_backward_activity_median_sustained_zero = np.where(is_backward_activity_median_zero_rolling == window_size,True,False)


        short_sleep_border_offwrist_initial = np.where(np.logical_and(is_temperature_variance_sustained_high,is_activity_median_sustained_zero),1,0)

        self.short_sleep_border_forward_offwrist_initial = np.where(np.logical_and(is_temperature_variance_sustained_high,is_forward_activity_median_sustained_zero),1,0)

        self.short_sleep_border_backward_offwrist_initial = np.where(np.logical_and(is_temperature_variance_sustained_high,is_backward_activity_median_sustained_zero),1,0)


        sleep_border_offwrists = zero_sequences(1-short_sleep_border_offwrist_initial,minimum_length=0)
        if self.verbose > 1:
                print("sleep_border_offwrists\n",sleep_border_offwrists)

        sleep_border_offwrists_count = len(sleep_border_offwrists)
        if sleep_border_offwrists_count > 0:
                sbo = 1
                while sbo < sleep_border_offwrists_count:
                        gap = sleep_border_offwrists[sbo,0] - sleep_border_offwrists[sbo-1,1]
                        if gap <= 2*window_size:
                                sleep_border_offwrists[sbo-1,1] = sleep_border_offwrists[sbo,1]
                                sleep_border_offwrists = np.delete(sleep_border_offwrists,sbo,0)
                                sleep_border_offwrists_count = len(sleep_border_offwrists)
                        else:
                                sbo += 1
        sleep_border_offwrists_count = len(sleep_border_offwrists)

        if sleep_border_offwrists_count > 0:
                sbo = 1
                while sbo < sleep_border_offwrists_count:
                        gap = sleep_border_offwrists[sbo,0] - sleep_border_offwrists[sbo-1,1]
                        if gap <= 2*window_size:
                                sleep_border_offwrists[sbo-1,1] = sleep_border_offwrists[sbo,1]
                                sleep_border_offwrists = np.delete(sleep_border_offwrists,sbo,0)
                                sleep_border_offwrists_count = len(sleep_border_offwrists)
                        else:
                                sbo += 1
        sleep_border_offwrists_count = len(sleep_border_offwrists)


        if self.verbose > 1:
                sleep_border_offwrists_df = self.periods_to_df(sleep_border_offwrists)
                sleep_border_offwrists_df = self.add_datetime_stamps(sleep_border_offwrists_df)
                print("sleep_border_offwrists refined\n",sleep_border_offwrists_df)

        possible_windows = []
        sleep_period_count = len(self.sleep_periods)
        possible_window_timedelta = timedelta(hours=self.possible_window_hours)
        forbidden_windows = []
        for i in range(sleep_period_count):
                possible_windows.append([self.sleep_periods.at[i,"datetime_start"]-possible_window_timedelta,self.sleep_periods.at[i,"datetime_start"]+possible_window_timedelta])
                possible_windows.append([self.sleep_periods.at[i,"datetime_end"]-possible_window_timedelta,self.sleep_periods.at[i,"datetime_end"]+possible_window_timedelta])

                # if self.do_forbidden_zone:
                forbidden_start = self.sleep_periods.at[i,"datetime_start"]+possible_window_timedelta
                forbidden_end = self.sleep_periods.at[i,"datetime_end"]-possible_window_timedelta
                # print("forbidden_start",forbidden_start)
                # print("forbidden_end",forbidden_end)

                forbidden_start = self.sleep_periods.at[i,"start"]+self.possible_window_hours*self.epoch_hour
                forbidden_end = self.sleep_periods.at[i,"end"]-self.possible_window_hours*self.epoch_hour

                # print(datetime_distance(self.datetime_stamps,forbidden_start))
                # forbidden_start = np.argmin(np.absolute(datetime_distance(self.datetime_stamps,forbidden_start)))
                # forbidden_end = np.argmin(np.absolute(datetime_distance(self.datetime_stamps,forbidden_end)))
                # print("forbidden_start",forbidden_start)
                # print("forbidden_end",forbidden_end)

                # input()

                forbidden_windows.append([forbidden_start,forbidden_end])

        possible_window_count = len(possible_windows)
        possible_windows = np.array(possible_windows)
        if self.verbose:
                print("possible_windows refined\n",possible_windows)
                
        forbidden_df = self.periods_to_df(forbidden_windows)
        self.forbidden_df = self.add_datetime_stamps(forbidden_df)
        if self.verbose:
                print("forbidden_windows")
                print(self.forbidden_df)

        self.forbidden_zone = np.zeros(self.data_length)
        for zone in forbidden_windows:
                self.forbidden_zone[zone[0]:zone[1]] = 1.0

        sbo = 0
        possible_window = 0
        # if self.verbose > 1:
        #         print("looping")
        while (sbo < sleep_border_offwrists_count) and (possible_window < possible_window_count):
                # if self.verbose > 1:
                #         print("current sleep_border_offwrist",sbo,"length",sleep_border_offwrists_count)
                #         print("current sleep_border_offwrist_df\n"+str(sleep_border_offwrists_df.loc[sbo,:]))
                #         print("possible_window ",possible_window)
                #         print(possible_windows[possible_window],"\n")

                remove = False
                if self.datetime_stamps[sleep_border_offwrists[sbo,1]] <= possible_windows[possible_window,1]:
                        if self.datetime_stamps[sleep_border_offwrists[sbo,0]] >= possible_windows[possible_window,0]:
                                sbo += 1   
                        else:
                                remove = True
                else:
                        if self.datetime_stamps[sleep_border_offwrists[sbo,0]] >= possible_windows[possible_window,1]:
                                possible_window += 1
                                if possible_window == possible_window_count:
                                        remove = True
                        else:
                                remove = True

                if remove:
                        sleep_border_offwrists = np.delete(sleep_border_offwrists,sbo,0)
                        sleep_border_offwrists_count = len(sleep_border_offwrists)
                        sleep_border_offwrists_df = self.periods_to_df(sleep_border_offwrists)
                        sleep_border_offwrists_df = self.add_datetime_stamps(sleep_border_offwrists_df)

        sleep_border_offwrists_count = len(sleep_border_offwrists)

        if self.verbose > 1:
                print("sleep_border_offwrists possible\n",sleep_border_offwrists_df)

        for i in range(sleep_border_offwrists_count):
                around_start = sleep_border_offwrists[i,0]-window_size
                around_end = sleep_border_offwrists[i,0]+half_window_size
                if around_start < 0: 
                        around_start = 0
                if around_end > self.data_length:
                        around_end = self.data_length

                highest_variance_around_index = np.argmax(self.normalized_temperature_variance[around_start:around_end])
                sleep_border_offwrists[i,0] = around_start + highest_variance_around_index

                around_start = sleep_border_offwrists[i,1]-half_window_size
                around_end = sleep_border_offwrists[i,1]+window_size
                if around_start < 0: 
                        around_start = 0
                if around_end > self.data_length:
                        around_end = self.data_length

                highest_variance_around_index = np.argmax(self.normalized_temperature_variance[around_start:around_end])
                sleep_border_offwrists[i,1] = around_start + highest_variance_around_index

        sleep_border_offwrists_count = len(sleep_border_offwrists)
        sleep_border_offwrists_df = self.periods_to_df(sleep_border_offwrists)
        sleep_border_offwrists_df = self.add_datetime_stamps(sleep_border_offwrists_df)

        if self.verbose > 1:
                print("sleep_border_offwrists peak\n",sleep_border_offwrists_df)

        i = 0
        while i < sleep_border_offwrists_count:
                length = sleep_border_offwrists[i,1] - sleep_border_offwrists[i,0]
                if length >= minimum_sleep_border_offwrist_length:
                        i += 1
                else:
                        sleep_border_offwrists = np.delete(sleep_border_offwrists,i,0)
                        sleep_border_offwrists_count = len(sleep_border_offwrists)

        sleep_border_offwrists_count = len(sleep_border_offwrists)

        if not self.do_analyze_sleep_borders:
                sleep_border_offwrists = []

        sleep_border_offwrists_df = self.periods_to_df(sleep_border_offwrists)
        sleep_border_offwrists_df = self.add_datetime_stamps(sleep_border_offwrists_df)
        sleep_border_offwrists_df["sleep_border"] = True
        self.sleep_border_offwrists_df = sleep_border_offwrists_df

        # if self.verbose > 1:
        #         print("sleep_border_offwrists final\n",self.sleep_border_offwrists_df)

        short_sleep_border_offwrist = np.zeros(self.data_length)
        for offwrist in sleep_border_offwrists:
                short_sleep_border_offwrist[offwrist[0]:offwrist[1]] = 1

        self.sleep_border_offwrists = sleep_border_offwrists
        self.short_sleep_border_offwrist = short_sleep_border_offwrist


    def compute_temperature_variations(self,offwrists_df):
        """Uses statistical measures to assert if the input data actually can
        be interpreted as having offwrist and onwrist periods.

        Returns
        -------
        is_bimodal : boolean
                If True, the input data can be mathematically interpreted as
                a random variables drawn from a bimodal distributions. The mo-
                des represent offwrist and onwrist periods.
        is_low_activity : boolean
                If True, the input data only has low activity values, indica-
                ting that the actigraph was never worn.
        """

        offwrists_df["decrease_ratio"] = 0.0
        # offwrists_df["temperature_increase"] = 0.0
        # offwrists_df["temperature_decrease"] = 0.0

        for offwrist in offwrists_df.index:
                start = offwrists_df.at[offwrist,"start"]
                end = offwrists_df.at[offwrist,"end"]

                window_temperature = self.temperature[start:end]
                window_temperature_derivative = np.array(self.temperature_derivative[start:end])

                plot_detail = False
                if plot_detail:
                        window_datetime = self.datetime_stamps[start:end]
                        plt.figure()
                        plt.plot(window_datetime,scale_by_max(window_activity),label="activity")
                        plt.plot(window_datetime,scale_by_max(window_temperature),label="temperature")
                        scaled_derivative = scale_by_max(window_temperature_derivative)

                new_vp = True
                if new_vp:
                        temperature_increase = np.where(window_temperature_derivative > 0)[0]
                        # temperature_increase = np.sum(window_temperature_derivative[temperature_increase])
                        if len(temperature_increase) > 2:
                                if plot_detail:
                                        plt.scatter(window_datetime[temperature_increase[0:-1]],scaled_derivative[temperature_increase[0:-1]],label="increase")
                                temperature_increase = np.sum(window_temperature_derivative[temperature_increase[1:-1]])
                        elif len(temperature_increase) == 2:
                                if plot_detail:
                                        plt.scatter(window_datetime[temperature_increase[0]],scaled_derivative[temperature_increase[0]],label="increase")
                                temperature_increase = np.sum(window_temperature_derivative[temperature_increase[1]])
                        else:
                                if plot_detail:
                                        plt.scatter(window_datetime[temperature_increase],scaled_derivative[temperature_increase],label="increase")
                                temperature_increase = np.sum(window_temperature_derivative[temperature_increase])

                        temperature_decrease = np.where(window_temperature_derivative < 0)[0]
                        # temperature_decrease = -1.0*np.sum(window_temperature_derivative[temperature_decrease]
                        if len(temperature_decrease) > 2:
                                if plot_detail:
                                        plt.scatter(window_datetime[temperature_decrease[1:]],scaled_derivative[temperature_decrease[1:]],label="decrease")
                                temperature_decrease = -1.0*np.sum(window_temperature_derivative[temperature_decrease[1:]])
                        elif len(temperature_decrease) == 2:
                                if plot_detail:
                                        plt.scatter(window_datetime[temperature_decrease[1]],scaled_derivative[temperature_decrease[1]],label="decrease")
                                temperature_decrease = -1.0*np.sum(window_temperature_derivative[temperature_decrease[1]])
                        else:
                                if plot_detail:
                                        plt.scatter(window_datetime[temperature_decrease],scaled_derivative[temperature_decrease],label="decrease")
                                temperature_decrease = -1.0*np.sum(window_temperature_derivative[temperature_decrease])

                else:
                        temperature_increase = np.where(window_temperature_derivative > 0)[0]
                        # temperature_increase = np.sum(window_temperature_derivative[temperature_increase])
                        # print("len(temperature_increase)",len(temperature_increase))
                        if len(temperature_increase) > 0:
                                if plot_detail:
                                        plt.scatter(window_datetime[temperature_increase],scaled_derivative[temperature_increase],label="increase")
                                temperature_increase = np.sum(window_temperature_derivative[temperature_increase])
                        else:
                                temperature_increase = 0

                        temperature_decrease = np.where(window_temperature_derivative < 0)[0]
                        # print("temperature_decrease",temperature_decrease)
                        # print("len(temperature_decrease)",len(temperature_decrease))
                        # input()
                        # temperature_decrease = -1.0*np.sum(window_temperature_derivative[temperature_decrease]
                        if len(temperature_decrease) > 0:
                                if plot_detail:
                                        plt.scatter(window_datetime[temperature_decrease],scaled_derivative[temperature_decrease],label="decrease")
                                temperature_decrease = -1.0*np.sum(window_temperature_derivative[temperature_decrease])
                        else:
                                temperature_decrease = 0

                if plot_detail:
                        plt.legend()
                        plt.grid()
                        plt.show()

                if temperature_increase > 0:
                     decrease_ratio = temperature_decrease/temperature_increase
                else:
                     decrease_ratio = temperature_decrease
                
                # offwrists_df.at[offwrist,"temperature_increase"] = temperature_increase
                # offwrists_df.at[offwrist,"temperature_decrease"] = temperature_decrease
                offwrists_df.at[offwrist,"decrease_ratio"] = decrease_ratio

        return offwrists_df
    

    def test_capsensor_statistically_above(self,peak_start,peak_end):
                offwrist_length = peak_end - peak_start
                capsensor_before = self.additional_information["c1"][peak_start-offwrist_length:peak_start]
                capsensor_after = self.additional_information["c1"][peak_end:peak_end+offwrist_length]
                capsensor_offwrist = self.additional_information["c1"][peak_start:peak_end]
                capsensor1_std = np.std(capsensor_offwrist)
                capsensor1_median = np.median(capsensor_offwrist)

                description_before = pd.Series(capsensor_before).describe()
                description_after = pd.Series(capsensor_after).describe()
                description_offwrist = pd.Series(capsensor_offwrist).describe()
                
                if self.verbose > 1:
                        print("short offwrist")
                        print("description_before1")
                        print(description_before)
                        print("description_after1")
                        print(description_after)
                        print("description_offwrist1")
                        print(description_offwrist)
                
                test_result_before_sensor1 = stats.kstest(capsensor_before,capsensor_offwrist,alternative="greater")
                test_result_after_sensor1 = stats.kstest(capsensor_after,capsensor_offwrist,alternative="greater")

                capsensor_before = self.additional_information["c2"][peak_start-offwrist_length:peak_start]
                capsensor_after = self.additional_information["c2"][peak_end:peak_end+offwrist_length]
                capsensor_offwrist = self.additional_information["c2"][peak_start:peak_end]
                capsensor2_std = np.std(capsensor_offwrist)
                capsensor2_median = np.median(capsensor_offwrist)

                description_before = pd.Series(capsensor_before).describe()
                description_after = pd.Series(capsensor_after).describe()
                description_offwrist = pd.Series(capsensor_offwrist).describe()
                
                test_result_before_sensor2 = stats.kstest(capsensor_before,capsensor_offwrist,alternative="greater")
                test_result_after_sensor2 = stats.kstest(capsensor_after,capsensor_offwrist,alternative="greater")

                pvalue_sum = test_result_after_sensor1.pvalue+test_result_before_sensor1.pvalue+test_result_after_sensor2.pvalue+test_result_before_sensor2.pvalue
                if self.verbose > 1:
                        print("description_before2")
                        print(description_before)
                        print("description_after2")
                        print(description_after)
                        print("description_offwrist2")
                        print(description_offwrist)

                        print("KS before1: ",test_result_before_sensor1)
                        print("KS after1: ",test_result_after_sensor1)
                        print("KS before2: ",test_result_before_sensor2)
                        print("KS after2: ",test_result_after_sensor2)
                        print("pvalue_sum: ",pvalue_sum)

                return pvalue_sum, capsensor1_std, capsensor1_median, capsensor2_std, capsensor2_median
    
    def test_temperature_difference_median_statistically_below(self,peak_start,peak_end):
                offwrist_length = peak_end - peak_start
                temperature_difference_before = self.additional_information["dif"][peak_start-offwrist_length:peak_start]
                temperature_difference_after = self.additional_information["dif"][peak_end:peak_end+offwrist_length]
                temperature_difference_offwrist = self.additional_information["dif"][peak_start:peak_end]
                temperature_difference_std = np.std(temperature_difference_offwrist)
                temperature_difference_median = np.median(temperature_difference_offwrist)

                description_before = pd.Series(temperature_difference_before).describe()
                description_after = pd.Series(temperature_difference_after).describe()
                description_offwrist = pd.Series(temperature_difference_offwrist).describe()
                
                if self.verbose > 1:
                        print("short offwrist")
                        print("description_before")
                        print(description_before)
                        print("description_after")
                        print(description_after)
                        print("description_offwrist")
                        print(description_offwrist)
                
                test_result_before = stats.kstest(temperature_difference_before,temperature_difference_offwrist,alternative="less")
                test_result_after = stats.kstest(temperature_difference_after,temperature_difference_offwrist,alternative="less")

                pvalue_sum = test_result_after.pvalue+test_result_before.pvalue
                if self.verbose > 1:
                        print("KS before: ",test_result_before)
                        print("KS after: ",test_result_after)

                        print("pvalue_sum: ",pvalue_sum)

                return pvalue_sum, temperature_difference_std, temperature_difference_median


    def valley_peak_detection(self,signal,index_shift,valley_quantile_threshold=0.98,peak_quantile_threshold=0.98,include_static=False,static_quantile=0.85):
        """Uses statistical measures to assert if the input data actually can
        be interpreted as having offwrist and onwrist periods.


        Returns
        -------
        is_bimodal : boolean
                If True, the input data can be mathematically interpreted as
                a random variables drawn from a bimodal distributions. The mo-
                des represent offwrist and onwrist periods.
        is_low_activity : boolean
                If True, the input data only has low activity values, indica-
                ting that the actigraph was never worn.
        """

        data_length = len(signal)
        
        positive_signal = signal[np.where(signal > 0)]
        positive_signal_indexes = np.arange(data_length)[np.where(signal > 0)]

        negative_signal = signal[np.where(signal < 0)]
        negative_signal_indexes = np.arange(data_length)[np.where(signal < 0)]

        if len(positive_signal) > 0:
                high_signal_level = np.quantile(positive_signal,peak_quantile_threshold,method='inverted_cdf')
                if include_static:
                        lower_high_signal_level = np.quantile(positive_signal,static_quantile,method='inverted_cdf')
                        signal_peaks, _ = find_peaks(positive_signal, height=lower_high_signal_level)

                        refined_peaks = []
                        for peak in signal_peaks:
                                if positive_signal[peak] >= high_signal_level:
                                        refined_peaks.append(peak)
                                else:
                                        signal_peak = positive_signal_indexes[peak]+index_shift
                                        if signal_peak >= self.minimum_offwrist_length:
                                                activity_before = self.activity[signal_peak-self.minimum_offwrist_length:signal_peak]
                                                
                                                low_activity_proportion = below_prop(activity_before,self.low_activity_threshold)
                                                low_activity_proportion = below_prop(activity_before,self.activity_threshold)
                                                
                                                zero_proportion = zero_prop(activity_before)

                                                if (low_activity_proportion >= 0.7) and (zero_proportion >= 0.4):
                                                        refined_peaks.append(peak)

                        signal_peaks = positive_signal_indexes[refined_peaks]
                else:
                        signal_peaks, _ = find_peaks(positive_signal, height=high_signal_level)
                        signal_peaks = positive_signal_indexes[signal_peaks]
        else:
              high_signal_level = 0
              signal_peaks = np.array([])

        if len(negative_signal) > 0:
                low_signal_level = np.quantile(-1*negative_signal,valley_quantile_threshold,method='inverted_cdf')
                if include_static:
                        higher_low_signal_level = np.quantile(-1*negative_signal,static_quantile,method='inverted_cdf')
                        signal_valleys, _ = find_peaks(-1*negative_signal,height=higher_low_signal_level)

                        refined_valleys = []
                        for valley in signal_valleys:
                                if negative_signal[valley] <= -low_signal_level:
                                        refined_valleys.append(valley)
                                else:
                                        signal_valley = negative_signal_indexes[valley]+index_shift
                                        if signal_valley <= self.data_length-self.minimum_offwrist_length:
                                                activity_after = self.activity[signal_valley:signal_valley+self.minimum_offwrist_length]
                                                
                                                low_activity_proportion = below_prop(activity_after,self.low_activity_threshold)
                                                low_activity_proportion = below_prop(activity_after,self.activity_threshold)

                                                zero_proportion = zero_prop(activity_after)

                                                if self.verbose:
                                                        print("trying to include static")
                                                        print("signal_valley",self.datetime_stamps[signal_valley])
                                                        print("low_activity_proportion",low_activity_proportion," zero_proportion",zero_proportion)

                                                if ( ((low_activity_proportion >= 0.7) and (zero_proportion >= 0.4)) or
                                                     (low_activity_proportion >= 0.9)
                                                   ):
                                                        refined_valleys.append(valley)

                        signal_valleys = negative_signal_indexes[refined_valleys]
                else:
                        signal_valleys, _ = find_peaks(-1*negative_signal, height=low_signal_level)
                        signal_valleys = negative_signal_indexes[signal_valleys]

        else:
              low_signal_level = 0
              signal_valleys = np.array([])

        return positive_signal, positive_signal_indexes, negative_signal, negative_signal_indexes, signal_peaks, signal_valleys

    def valley_peak_offwrist_algorithm(self,include_static=False,static_quantile=0.92):
        """Uses statistical measures to assert if the input data actually can
        be interpreted as having offwrist and onwrist periods.


        Returns
        -------
        is_bimodal : boolean
                If True, the input data can be mathematically interpreted as
                a random variables drawn from a bimodal distributions. The mo-
                des represent offwrist and onwrist periods.
        is_low_activity : boolean
                If True, the input data only has low activity values, indica-
                ting that the actigraph was never worn.
        """

        day_split_temperature_derivative,dates = actigraphy_split_by_day(self.temperature_derivative,self.datetime_stamps)
        if self.verbose:
                print("day_split_temperature_derivative")
                print(day_split_temperature_derivative)
                print("dates")
                print(dates)

        temperature_derivative_peaks = []
        temperature_derivative_valleys = []

        date_index = 0
        index_shift = 0
        for day in day_split_temperature_derivative:
                self.positive_temperature_derivative, self.positive_temperature_derivative_indexes, self.negative_temperature_derivative, self.negative_temperature_derivative_indexes, day_temperature_derivative_peaks, day_temperature_derivative_valleys = self.valley_peak_detection(day,index_shift,self.valley_quantile,self.peak_quantile,include_static,static_quantile)
                day_temperature_derivative_peaks = day_temperature_derivative_peaks + index_shift
                day_temperature_derivative_valleys = day_temperature_derivative_valleys + index_shift

                temperature_derivative_peaks = np.hstack((temperature_derivative_peaks,day_temperature_derivative_peaks))
                temperature_derivative_valleys = np.hstack((temperature_derivative_valleys,day_temperature_derivative_valleys))

                # print("information for day", dates[date_index])
                # print("self.temperature_derivative",day)
                # print("positive_temperature_derivative",self.positive_temperature_derivative)
                # print("positive_temperature_derivative_indexes",self.positive_temperature_derivative_indexes)
                # print("negative_temperature_derivative",self.negative_temperature_derivative)
                # print("negative_temperature_derivative_indexes",self.negative_temperature_derivative_indexes)
                # print("day_temperature_derivative_peaks",day_temperature_derivative_peaks)
                # print("day_temperature_derivative_valleys",day_temperature_derivative_valleys)
                # print("temperature_derivative_peaks",temperature_derivative_peaks)
                # print("temperature_derivative_valleys",temperature_derivative_valleys)
                # input()

                date_index += 1
                index_shift += len(day)

        temperature_derivative_peaks = temperature_derivative_peaks.astype(int)
        temperature_derivative_valleys = temperature_derivative_valleys.astype(int)

        if self.verbose:
                print("temperature_derivative_peaks",temperature_derivative_peaks)
                print("temperature_derivative_valleys",temperature_derivative_valleys)
        # input()

        self.temperature_derivative_peaks = temperature_derivative_peaks
        self.temperature_derivative_valleys = temperature_derivative_valleys

        peak_count = len(temperature_derivative_peaks)
        valley_count = len(temperature_derivative_valleys)

        valley_peak_offwrists = []
        valley_peak_offwrists_df = self.periods_to_df(valley_peak_offwrists)
        valley_peak_offwrists_df = self.add_datetime_stamps(valley_peak_offwrists_df)
        valley = 0
        peak = 0


        first_peak = temperature_derivative_peaks[peak]
        first_valley = temperature_derivative_valleys[valley]
        if self.verbose:
                print("first_peak",self.datetime_stamps[first_peak])
                print("first_valley",self.datetime_stamps[first_valley])

        if first_peak < first_valley:
                if self.verbose:
                        print("first_peak prop",below_prop(self.activity[0:first_peak],self.activity_threshold))
                length = first_peak
                if ((length >= self.minimum_offwrist_length) and  (below_prop(self.activity[0:first_peak],self.activity_threshold) > 0.75)):
                        valley_peak_offwrists = [[0,first_peak]]
                        valley_peak_offwrists_df = self.periods_to_df(valley_peak_offwrists)
                        valley_peak_offwrists_df = self.add_datetime_stamps(valley_peak_offwrists_df)

        


        # if self.verbose > 1:
        #         print("searching v-p")
        while (valley < valley_count) and (peak < peak_count):
                # if self.verbose > 1:
                #         print("current valley",valley,temperature_derivative_valleys[valley])
                #         print("current peak",peak,temperature_derivative_peaks[peak])
                        
                #         print("valley_peak_offwrists")
                #         print(valley_peak_offwrists_df)
                        # input()

                remove = False
                if temperature_derivative_peaks[peak] > temperature_derivative_valleys[valley]:
                        length = temperature_derivative_peaks[peak] - temperature_derivative_valleys[valley]

                        if valley+1 < valley_count:
                                next_possible_length = self.minimum_offwrist_length-1                            
                                if self.next_possible_length_criteria:
                                        next_possible_length = temperature_derivative_peaks[peak] - temperature_derivative_valleys[valley+1]
                                if (temperature_derivative_peaks[peak] > temperature_derivative_valleys[valley+1]) and (next_possible_length >= self.minimum_offwrist_length):                                        
                                        valley += 1
                                else:
                                        if ((length >= self.minimum_offwrist_length) and (length < self.long_offwrist_length)
                                                ):
                                                # if temperature_derivative_peaks[peak]+1 < self.data_length:
                                                #         valley_peak_offwrists.append([temperature_derivative_valleys[valley],temperature_derivative_peaks[peak]])
                                                # else:
                                                valley_peak_offwrists.append([temperature_derivative_valleys[valley],temperature_derivative_peaks[peak]])
                                                valley_peak_offwrists_df = self.periods_to_df(valley_peak_offwrists)
                                                valley_peak_offwrists_df = self.add_datetime_stamps(valley_peak_offwrists_df)
                                                peak += 1
                                        
                                        valley += 1


                        else:
                                if ((length >= self.minimum_offwrist_length) and (length < self.long_offwrist_length)
                                        ):
                                        # if temperature_derivative_peaks[peak]+1 < self.data_length:
                                        #         valley_peak_offwrists.append([temperature_derivative_valleys[valley],temperature_derivative_peaks[peak]])
                                        # else:
                                        valley_peak_offwrists.append([temperature_derivative_valleys[valley],temperature_derivative_peaks[peak]])

                                        valley_peak_offwrists_df = self.periods_to_df(valley_peak_offwrists)
                                        valley_peak_offwrists_df = self.add_datetime_stamps(valley_peak_offwrists_df)
                                        peak += 1

                                valley += 1
                else:
                        peak += 1

        last_peak = temperature_derivative_peaks[peak_count-1]
        last_valley = temperature_derivative_valleys[valley_count-1]
        if last_peak < last_valley:
                if self.verbose:
                        print("last_valley prop",below_prop(self.activity[last_valley:self.data_length],self.activity_threshold))
                        
                length = self.data_length - last_valley
                if ((length >= self.minimum_offwrist_length) and  (below_prop(self.activity[last_valley:self.data_length],self.activity_threshold) > 0.75)):
                        valley_peak_offwrists.append([last_valley,self.data_length])
                        valley_peak_offwrists_df = self.periods_to_df(valley_peak_offwrists)
                        valley_peak_offwrists_df = self.add_datetime_stamps(valley_peak_offwrists_df)

        if self.verbose:
                print("last_peak",self.datetime_stamps[last_peak])
                print("last_valley",self.datetime_stamps[last_valley])

        
        raw_valley_peak_detection = np.ones(self.data_length)
        for offwrist in valley_peak_offwrists_df.index:
                start = valley_peak_offwrists_df.at[offwrist,"start"]
                end = valley_peak_offwrists_df.at[offwrist,"end"]

                raw_valley_peak_detection[start:end] = 0.0
        
        self.raw_valley_peak_detection = raw_valley_peak_detection
        self.raw_valley_peak_detection_df = valley_peak_offwrists_df

        self.forbidden_filtered = False
        if self.do_forbidden_zone:
                allowed_valley_peak_offwrists = []
                filtered_valley_peak_offwrists = []
                offwrist_index_number = 0
                for offwrist_index in valley_peak_offwrists:
                    forbidden = self.forbidden_zone[offwrist_index[0]:offwrist_index[1]]
                    offwrist_length = offwrist_index[1] - offwrist_index[0]
                    forbidden_epochs = np.sum(forbidden)
                    if forbidden_epochs == 0:
                        allowed_valley_peak_offwrists.append(offwrist_index)
                    else:
                        if offwrist_length > self.possible_window_hours:
                                allowed_sequence = zero_sequences(forbidden)
                                if len(allowed_sequence) > 0:
                                        forbidden_length = allowed_sequence[0,1] - allowed_sequence[0,0]
                                        distance_from_transition = self.possible_window_hours - forbidden_length
                                        if distance_from_transition <= 0.66*self.possible_window_hours:
                                                allowed_valley_peak_offwrists.append(offwrist_index)
                                        else:
                                                filtered_valley_peak_offwrists.append(offwrist_index_number)
                                else:
                                        filtered_valley_peak_offwrists.append(offwrist_index_number)

                    offwrist_index_number += 1

                valley_peak_offwrists_count = len(valley_peak_offwrists)
                valley_peak_offwrists = allowed_valley_peak_offwrists

                if self.verbose:
                        print("all valley peak")
                        print(valley_peak_offwrists_df)
                        if len(filtered_valley_peak_offwrists) > 0:
                                print("forbidden valley peak")
                                print(valley_peak_offwrists_df.loc[filtered_valley_peak_offwrists,:])

                if len(valley_peak_offwrists) < valley_peak_offwrists_count:
                        self.forbidden_filtered = True
                        valley_peak_offwrists_df = self.periods_to_df(valley_peak_offwrists)
                        valley_peak_offwrists_df = self.add_datetime_stamps(valley_peak_offwrists_df)

                if self.verbose:
                        print("allowed valley peak")
                        print(valley_peak_offwrists_df)

        valley_peak_offwrists_count = len(valley_peak_offwrists)

        valley_peak_offwrists_df["length"] = valley_peak_offwrists_df["end"] - valley_peak_offwrists_df["start"]
        valley_peak_offwrists_df["low_activity_proportion"] = 0.0
        valley_peak_offwrists_df["zero_activity_proportion"] = 0.0
        valley_peak_offwrists_df["low_temperature_proportion"] = 0.0
        # valley_peak_offwrists_df["overlaps_forward"] = False
        # valley_peak_offwrists_df["overlaps_backward"] = False
        for offwrist in valley_peak_offwrists_df.index:
                start = valley_peak_offwrists_df.at[offwrist,"start"]
                end = valley_peak_offwrists_df.at[offwrist,"end"]

                window_activity = self.activity[start:end]
                window_forward_offwrist = self.short_sleep_border_forward_offwrist_initial[start:end]
                window_backward_offwrist = self.short_sleep_border_backward_offwrist_initial[start:end]
                window_temperature = self.temperature[start:end]

                # positive_activity = window_activity[np.where(window_activity > 0,True,False)]
                low_activity_proportion = below_prop(window_activity,self.low_activity_threshold)
                low_activity_proportion = below_prop(window_activity,self.activity_threshold)
                zero_activity_proportion = zero_prop(window_activity)

                low_temperature_proportion = below_prop(window_temperature,self.temperature_threshold)

                overlaps_forward = np.sum(window_forward_offwrist)
                overlaps_backward = np.sum(window_backward_offwrist)
                
                valley_peak_offwrists_df.at[offwrist,"low_activity_proportion"] = low_activity_proportion
                valley_peak_offwrists_df.at[offwrist,"low_temperature_proportion"] = low_temperature_proportion
                valley_peak_offwrists_df.at[offwrist,"zero_activity_proportion"] = zero_activity_proportion
                # valley_peak_offwrists_df.at[offwrist,"overlaps_forward"] = (overlaps_forward > 0)
                # valley_peak_offwrists_df.at[offwrist,"overlaps_backward"] = (overlaps_backward > 0)

                if not self.is_lumus_file:
                        window_temperature_difference = self.additional_information["dif"][start:end]
                        valley_peak_offwrists_df.at[offwrist,"temperature_difference_median"] = np.median(window_temperature_difference)

        # valley_peak_offwrists_df["overlaps"] = np.logical_or(valley_peak_offwrists_df["overlaps_forward"],valley_peak_offwrists_df["overlaps_backward"])
        valley_peak_offwrists_df = self.compute_temperature_variations(valley_peak_offwrists_df)
        
        if self.verbose:
                print("initial valley peak sleep_border_offwrists")
                print(valley_peak_offwrists_df)

        do_valley_peak_low_activity_filter = True
        valley_peak_low_activity_minimum = 0.5
        if do_valley_peak_low_activity_filter:
                valid_index = []
                for offwrist in valley_peak_offwrists_df.index:
                     if valley_peak_offwrists_df.at[offwrist,"low_activity_proportion"] > valley_peak_low_activity_minimum:
                          valid_index.append(offwrist)
                #      elif valley_peak_offwrists_df.at[offwrist,"zero_activity_proportion"] >= 0.95:
                #                 valid_index.append(offwrist)

                valley_peak_offwrists_df = valley_peak_offwrists_df.loc[valid_index,:]
                valley_peak_offwrists_df.index = range(len(valley_peak_offwrists_df))
        
        if self.verbose:
                print("low act filtered valley peak sleep_border_offwrists")
                print(valley_peak_offwrists_df)

        do_valley_peak_low_temperature_filter = True

        # short_valley_peak_offwrist_criteria = True
        # medium_valley_peak_offwrist_criteria = True

        valley_peak_low_temperature_minimum = 0.5
        # short_offwrist_minimum_decrease_ratio = 1.25
        # lumus_short_offwrist_maximum_capsensor_pvalue = 5e-4
        # self.short_offwrist_length = 20
        if do_valley_peak_low_temperature_filter:
                valid_index = []
                for offwrist in valley_peak_offwrists_df.index:
                        valley_peak_low_temperature_proportion = valley_peak_offwrists_df.at[offwrist,"low_temperature_proportion"]
                        if valley_peak_low_temperature_proportion > valley_peak_low_temperature_minimum:
                                valid_index.append(offwrist)
                                
                        else:
                                if False:
                                # if valley_peak_offwrists_df.at[offwrist,"zero_activity_proportion"] >= 0.9:
                                #         valid_index.append(offwrist)
                                        pass
                                
                                else:
                                        offwrist_length = valley_peak_offwrists_df.at[offwrist,"length"]
                                        peak_start = valley_peak_offwrists_df.at[offwrist,"start"]
                                        peak_end = valley_peak_offwrists_df.at[offwrist,"end"]
                                        
                                        forbidden = self.forbidden_zone[peak_start:peak_end]
                                        if self.short_valley_peak_offwrist_criteria and (offwrist_length <= self.short_offwrist_length):
                                                if self.verbose > 1:
                                                        print("short offwrist")
                                                        print(valley_peak_offwrists_df.loc[offwrist,:])                                                

                                                if np.sum(forbidden) == 0:
                                                        low_temperature_epochs = int(round(valley_peak_low_temperature_proportion*valley_peak_offwrists_df.at[offwrist,"length"]))
                                                        if (valley_peak_offwrists_df.at[offwrist,"decrease_ratio"] > self.short_offwrist_minimum_decrease_ratio):
                                                                if (low_temperature_epochs > 1):
                                                                        valid_index.append(offwrist)
                                                                else:
                                                                        if self.is_lumus_file:
                                                                                pvalue_sum, capsensor1_std, capsensor1_median, capsensor2_std, capsensor2_median = self.test_capsensor_statistically_above(peak_start,peak_end)
                                                                                if ( (pvalue_sum < self.lumus_short_offwrist_maximum_capsensor_pvalue) and 
                                                                                ( 
                                                                                ( (capsensor1_std < self.offwrist_maximum_capsensor1_std) and (capsensor2_std < self.offwrist_maximum_capsensor2_std) ) and
                                                                                ( (capsensor1_median > self.offwrist_minimum_capsensor1_median) and (capsensor2_median > self.offwrist_minimum_capsensor2_median) )
                                                                                )
                                                                                ):
                                                                                        valid_index.append(offwrist)
                                                                        else:
                                                                                pvalue_sum, temperature_difference_std, temperature_difference_median = self.test_temperature_difference_median_statistically_below(peak_start,peak_end)
                                                                                if ( (pvalue_sum < self.trust_short_offwrist_maximum_temperature_difference_pvalue) and 
                                                                                ( 
                                                                                (temperature_difference_median < self.offwrist_minimum_temperature_difference_median-0.1)
                                                                                )
                                                                                ):
                                                                                        valid_index.append(offwrist)
                                                                                        

                                        elif self.medium_valley_peak_offwrist_criteria and (offwrist_length <= 2*self.short_offwrist_length):
                                                if self.verbose > 1:
                                                        print("offwrist_length <= 2*self.short_offwrist_length")
                                                        print(valley_peak_offwrists_df.loc[offwrist,:])   

                                                if ( (valley_peak_offwrists_df.at[offwrist,"decrease_ratio"] > self.short_offwrist_minimum_decrease_ratio) and 
                                                ( (valley_peak_low_temperature_proportion > 0.8*valley_peak_low_temperature_minimum) or ()
                                                )
                                                ):
                                                        valid_index.append(offwrist)

                valley_peak_offwrists_df = valley_peak_offwrists_df.loc[valid_index,:]
                valley_peak_offwrists_df.index = range(len(valley_peak_offwrists_df))

        if self.verbose:
                print("low temp filtered valley peak sleep_border_offwrists")
                print(valley_peak_offwrists_df)

        # do_valley_peak_overlaps_forward_filter = True
        # if do_valley_peak_overlaps_forward_filter:
        #         valley_peak_offwrists_df = valley_peak_offwrists_df[valley_peak_offwrists_df["overlaps_forward"] == True]
        #         valley_peak_offwrists_df.index = range(len(valley_peak_offwrists_df))
        # print("overlap forward filtered valley peak sleep_border_offwrists")
        # print(valley_peak_offwrists_df)

        # do_valley_peak_overlaps_backward_filter = True
        # if do_valley_peak_overlaps_backward_filter:
        #         valley_peak_offwrists_df = valley_peak_offwrists_df[valley_peak_offwrists_df["overlaps_backward"] == True]
                # valley_peak_offwrists_df.index = range(len(valley_peak_offwrists_df))
        # print("overlap backward filtered valley peak sleep_border_offwrists")
        # print(valley_peak_offwrists_df)

        # do_valley_peak_overlaps_forward_filter = True
        # if do_valley_peak_overlaps_forward_filter:
        #         valley_peak_offwrists_df = valley_peak_offwrists_df[valley_peak_offwrists_df["overlaps"] == True]
        #         valley_peak_offwrists_df.index = range(len(valley_peak_offwrists_df))
        # print("overlap filtered valley peak sleep_border_offwrists")
        # print(valley_peak_offwrists_df)

        do_valley_peak_decrease_ratio_filter = True
        # do_valley_peak_decrease_ratio_filter = False
        if do_valley_peak_decrease_ratio_filter:
                valid_index = []
                for offwrist in valley_peak_offwrists_df.index:
                        if valley_peak_offwrists_df.at[offwrist,"decrease_ratio"] > self.decrease_ratio_minimum:
                                valid_index.append(offwrist)
                                
                        # elif valley_peak_offwrists_df.at[offwrist,"zero_activity_proportion"] >= 0.95:
                        #         valid_index.append(offwrist)

                valley_peak_offwrists_df = valley_peak_offwrists_df.loc[valid_index,:]
                valley_peak_offwrists_df.index = range(len(valley_peak_offwrists_df))

                # valley_peak_offwrists_df = valley_peak_offwrists_df[(valley_peak_offwrists_df["decrease_ratio"] > self.decrease_ratio_minimum)]
                
                if self.verbose:
                        print("decrease ratio filtered valley peak sleep_border_offwrists")
                        print(valley_peak_offwrists_df)



        do_valley_peak_temperature_difference_filter = True
        # do_valley_peak_decrease_ratio_filter = False
        if ((not self.is_lumus_file) and do_valley_peak_temperature_difference_filter):
                valid_index = []
                for offwrist in valley_peak_offwrists_df.index:
                        if valley_peak_offwrists_df.at[offwrist,"temperature_difference_median"] < self.offwrist_maximum_temperature_difference_median:
                                valid_index.append(offwrist)
                                
                        # elif valley_peak_offwrists_df.at[offwrist,"zero_activity_proportion"] >= 0.95:
                        #         valid_index.append(offwrist)

                valley_peak_offwrists_df = valley_peak_offwrists_df.loc[valid_index,:]
                valley_peak_offwrists_df.index = range(len(valley_peak_offwrists_df))

                if self.verbose:
                        print("temperature difference filtered valley peak sleep_border_offwrists")
                        print(valley_peak_offwrists_df)


        self.computed_valley_peak_offwrists_df = valley_peak_offwrists_df
        if self.do_valley_peak_algorithm:
                self.valley_peak_offwrists_df = valley_peak_offwrists_df
        else:
                self.valley_peak_offwrists_df = self.periods_to_df([])

        self.computed_valley_peak_offwrists = self.df_to_periods(self.computed_valley_peak_offwrists_df)
        self.valley_peak_offwrists = self.df_to_periods(self.valley_peak_offwrists_df)
        # input()

        computed_valley_peak_offwrist = np.zeros(self.data_length)
        for offwrist in self.computed_valley_peak_offwrists:
                computed_valley_peak_offwrist[offwrist[0]:offwrist[1]] = 1

        valley_peak_offwrist = np.zeros(self.data_length)
        for offwrist in self.valley_peak_offwrists:
                valley_peak_offwrist[offwrist[0]:offwrist[1]] = 1

        self.computed_valley_peak_offwrist = computed_valley_peak_offwrist
        self.valley_peak_offwrist = valley_peak_offwrist


    def surrounded_valley_peak_detection(self,):
        """Uses statistical measures to assert if the input data actually can
        be interpreted as having offwrist and onwrist periods.


        Returns
        -------
        is_bimodal : boolean
                If True, the input data can be mathematically interpreted as
                a random variables drawn from a bimodal distributions. The mo-
                des represent offwrist and onwrist periods.
        is_low_activity : boolean
                If True, the input data only has low activity values, indica-
                ting that the actigraph was never worn.
        """

        self.raw_valley_peak_detection_df["length"] = self.raw_valley_peak_detection_df["end"] - self.raw_valley_peak_detection_df["start"]
        # valley_peak_offwrists_df["low_activity_proportion"] = 0.0
        # valley_peak_offwrists_df["low_temperature_proportion"] = 0.0
        # valley_peak_offwrists_df["overlaps_forward"] = False
        # valley_peak_offwrists_df["overlaps_backward"] = False

        self.raw_valley_peak_detection_df = self.compute_temperature_variations(self.raw_valley_peak_detection_df)

        if self.verbose:
              print("self.raw_valley_peak_detection_df")
              print(self.raw_valley_peak_detection_df)

        detection_indexes = []
        surrounded_valley_peak_offwrist = np.ones(self.data_length)
        for offwrist in self.raw_valley_peak_detection_df.index:
                start = self.raw_valley_peak_detection_df.at[offwrist,"start"]
                end = self.raw_valley_peak_detection_df.at[offwrist,"end"]
                length = self.raw_valley_peak_detection_df.at[offwrist,"length"]

                preceding_high_activity_proportion = 1-below_prop(self.activity[start-self.minimum_offwrist_length:start],self.activity_threshold)
                succeding_high_activity_proportion = 1-below_prop(self.activity[end:end+self.minimum_offwrist_length],self.activity_threshold)
                low_activity_proportion = below_prop(self.activity[start:end],self.activity_threshold)

                if self.verbose:
                        print("start, end, length")
                        print(start, end, length)
                        print("preceding_high_activity_proportion",preceding_high_activity_proportion)
                        print("succeding_high_activity_proportion",succeding_high_activity_proportion)
                        print("low_activity_proportion",low_activity_proportion)

                if ((self.raw_valley_peak_detection_df.at[offwrist,"decrease_ratio"] > 1.0) and (low_activity_proportion > 0.75)):
                        if ((preceding_high_activity_proportion > 0.75) and (succeding_high_activity_proportion > 0.75)):
                                detection_indexes.append(offwrist)        
                                surrounded_valley_peak_offwrist[start:end] = 0.0

        surrounded_valley_peak_offwrist_df = self.raw_valley_peak_detection_df.loc[detection_indexes,:]

        self.surrounded_valley_peak_offwrist = surrounded_valley_peak_offwrist
        self.surrounded_valley_peak_offwrist_df = surrounded_valley_peak_offwrist_df


    def check_bimodality(self,):
        """Uses statistical measures to assert if the input data actually can
        be interpreted as having offwrist and onwrist periods.


        Returns
        -------
        is_bimodal : boolean
                If True, the input data can be mathematically interpreted as
                a random variables drawn from a bimodal distributions. The mo-
                des represent offwrist and onwrist periods.
        is_low_activity : boolean
                If True, the input data only has low activity values, indica-
                ting that the actigraph was never worn.
        """
        
        self.activity_median_high_proportion = zero_prop(self.activity_median_low)

        low_median_activity_bool = self.activity_median_low.astype(bool)
        self.low_activity_low_temperature_proportion = below_prop(self.temperature[low_median_activity_bool],self.temperature_threshold)

        positive_act = self.activity[np.where(self.activity > 0,True,False)]
        
        low_positive_activity_proportion = below_prop(positive_act,self.low_activity_threshold)
        low_positive_activity_proportion = below_prop(positive_act,self.activity_threshold)
        
        initial_offwrist_proportion = zero_prop(self.sleep_filtered_offwrist)

        # if self.dev:
        if self.verbose:
                print("activity_zero_proportion:",self.activity_zero_proportion)
                print("offwrist proportion:",initial_offwrist_proportion)
                print("low disc proportion:",1-self.activity_median_high_proportion)
                print("low temperature proportion:",self.low_temperature_proportion)
                print("low act low temperature proportion:",self.low_activity_low_temperature_proportion)
                print("low positive act:",low_positive_activity_proportion)

        self.temperature_mean = np.mean(self.temperature)

        self.initial_offwrist_proportion = initial_offwrist_proportion
        self.low_positive_activity_proportion = low_positive_activity_proportion

        is_low_activity = False
        is_bimodal = True
        # Low activity criteria is tested using low and zero activity propor-
        # tions.
        if (self.activity_zero_proportion >= 0.9) or (low_positive_activity_proportion > 0.8):
            is_bimodal = False
            is_low_activity = True
            if self.verbose:
                print("low act")
        else:
            # Bimodality is tested using the Ashman's D statistic. 
            if self.ashman_d <= self.ashman_d_minimum:
                if ((initial_offwrist_proportion >= self.bimodal_maximum_offwrist_proportion) and (low_positive_activity_proportion >= self.bimodal_minimum_low_activity_proportion)):
                        is_bimodal = False
                
                if self.do_temperature_criteria:
                        if (low_positive_activity_proportion > 0.3): #and (self.temperature_threshold > self.temperature_mean): # and (low_temperature_proportion > 0.5):
                                is_bimodal = False

        #     else:  
        #         if self.ashman_d < self.ashman_d_maximum:
        #                 # If the estimated temperature threshold that separates off-
        #                 # and onwrist states is above the temperature mean, it pro-
        #                 # bably indicates that there aren't any offwrist periods,
        #                 # because offwrist periods typically have a significantly
        #                 # lower temperature mean.
        #                 if  (low_positive_activity_proportion > 0.3) and (self.temperature_threshold > self.temperature_mean): # and (low_temperature_proportion > 0.5):
        #                         is_bimodal = False

        return is_bimodal,is_low_activity
    
    def onwrist_refinement(self,onwrist_periods):
        onwrist_refinement_window = self.minimum_onwrist_length
        print(onwrist_periods)
        
        onwrist_count = len(onwrist_periods)

        refined_onwrist_periods = [onwrist_periods[0]]
        
        onwrist_start = onwrist_periods[0,0]
        possible_border = onwrist_start-1
        
        if self.verbose > 1:
                print("self.sleep_low_activity_threshold",self.sleep_low_activity_threshold)
        border_search = True
        while border_search and (possible_border > onwrist_refinement_window):
                if self.verbose > 1:
                        print("border",self.datetime_stamps[possible_border])
                        print("border median acitivity",self.activity_median[possible_border])

                if self.is_lumus_file:
                        capsensor_before = self.additional_information["c1"][possible_border-onwrist_refinement_window:possible_border]
                        capsensor_after = self.additional_information["c1"][possible_border:possible_border+onwrist_refinement_window]

                        description_before = pd.Series(capsensor_before).describe()
                        description_after = pd.Series(capsensor_after).describe()
                        
                        test_result = stats.kstest(capsensor_before,capsensor_after,alternative="greater")
                        if self.verbose > 1:
                                print("description_before")
                                print(description_before)
                                print("description_after")
                                print(description_after)
                                print("KS: ",test_result)
                        

                if self.activity_median[possible_border] < 0.35*self.minimum_awake_median_activity_level:                        
                        if test_result.pvalue < 0.05:
                                if self.verbose > 1:
                                        print("is less")
                                border_search = False
                                refined_onwrist_periods[0,0] = possible_border
                        else:
                                if self.verbose > 1:
                                        print("not less")
                                possible_border -= 1
                else:   
                        possible_border -= 1
                
                input("descriptions")
              
        # if border_search and (possible_border == onwrist_refinement_window):

        # onwrist_end = onwrist_periods[0,1]
        
        # onwrist = 1
        # while onwrist < onwrist_count-1:

        
              
    def onwrist_border_refinement(self,onwrist_periods_df,refinement_region_border_length_hours=24):
        print("onwrist_border_refinement")
        # print("self.verbose",self.verbose)

        refined_offwrist = np.zeros(self.data_length)
        
        refinement_region_border_length = refinement_region_border_length_hours*self.epoch_hour

        onwrist_count = len(onwrist_periods_df)

        onwrist_refinement_regions = []

        onwrist = 0
        while onwrist < onwrist_count:
                onwrist_start = onwrist_periods_df.at[onwrist,"start"]
                onwrist_end = onwrist_periods_df.at[onwrist,"end"]

                refinement_region_start = 0
                if onwrist_start >= refinement_region_border_length:
                      refinement_region_start = onwrist_start-refinement_region_border_length

                if self.verbose > 1:
                      print("refinement_region_start",refinement_region_start)

                refinement_region_end = self.data_length
                while (onwrist+1 < onwrist_count) and (refinement_region_end == self.data_length):
                        next_onwrist_start = onwrist_periods_df.at[onwrist+1,"start"]
                        if next_onwrist_start > onwrist_end+refinement_region_border_length:
                              refinement_region_end = onwrist_end+refinement_region_border_length
                        else:
                              onwrist += 1
                              onwrist_end = onwrist_periods_df.at[onwrist,"end"]

                        if self.verbose > 1:
                                print("next_onwrist_start",next_onwrist_start)
                                print("refinement_region_end",refinement_region_end)

                if onwrist == onwrist_count-1:
                        if self.data_length > onwrist_end+refinement_region_border_length:
                                refinement_region_end = onwrist_end+refinement_region_border_length
                
                if self.verbose > 1:
                      print("refinement_region_end",refinement_region_end)

                onwrist_refinement_regions.append([refinement_region_start,refinement_region_end])
                if self.verbose > 1:
                      print("onwrist_refinement_regions",onwrist_refinement_regions)
                
                onwrist += 1
        
        offwrist_activity_quantile=0.15 # Quantile used to compute low activity level
        minimum_normalized_activity_threshold=0.015 # Minimum level of the normalized acitivity median to 
        for region in onwrist_refinement_regions:
                print("\n\ncropped region recursive refinement")
                region_start = region[0]
                region_end = region[1]

                datetime_stamps = self.datetime_stamps[region_start:region_end]
                temperature = self.temperature[region_start:region_end]
                activity = self.activity[region_start:region_end]
                activity_median = self.activity_median[region_start:region_end]
                temperature_derivative = self.temperature_derivative[region_start:region_end]
                temperature_derivative_variance = self.temperature_derivative_variance[region_start:region_end]
                normalized_temperature_variance = self.normalized_temperature_variance[region_start:region_end]
                normalized_activity_median = norm_01(activity_median)

                cropped_additional_information = {}
                for key, value in self.additional_information.items():
                        cropped_additional_information[key] = value[region_start:region_end]

                activity_median_zero_proportion = zero_prop(normalized_activity_median)
                low_activity_median_quantile = activity_median_zero_proportion + offwrist_activity_quantile*(1-activity_median_zero_proportion)
                low_normalized_activity_median_threshold = np.quantile(normalized_activity_median,low_activity_median_quantile,method='inverted_cdf')
                if low_normalized_activity_median_threshold < minimum_normalized_activity_threshold:
                        low_normalized_activity_median_threshold = minimum_normalized_activity_threshold

                is_normalized_activity_median_low_bool = np.where(normalized_activity_median < low_normalized_activity_median_threshold,True,False)
                is_normalized_activity_median_low_int = is_normalized_activity_median_low_bool.astype(int)

                normalized_temperature = norm_01(temperature)
                start_time = ttime.time()
                temperature_threshold, ashman_d = bimodal_thresh(normalized_temperature[is_normalized_activity_median_low_bool],nbins=100,plot=False,save_plot=False,title="temperature",verbose=True)
                thresh_time = ttime.time()-start_time
                if self.verbose > 1:
                        print("thresh_time",thresh_time)
                        print("ashman_d",ashman_d)

                # plt.figure()
                # plt.plot(datetime_stamps,norm_01(activity),label="activity")
                # plt.plot(datetime_stamps,norm_01(self.activity_median[region_start:region_end],normalize_like=activity),label="activity_median")
                # plt.plot(datetime_stamps,normalized_temperature,label="temperature")
                # plt.plot(datetime_stamps,temperature_threshold*np.ones(len(datetime_stamps)),label="threshold")

                median_high_activity_temperature = np.median(temperature[np.logical_not(is_normalized_activity_median_low_bool)])
                if self.verbose > 1:
                        print("median_high_activity_temperature",median_high_activity_temperature)

                # Rescaling normalized temperature threshold
                temperature_minimum = np.min(temperature) 
                temperature_maximum = np.max(temperature)
                temperature_threshold = temperature_minimum + temperature_threshold*(temperature_maximum - temperature_minimum)

                if temperature_threshold > median_high_activity_temperature:
                        temperature_threshold = median_high_activity_temperature
                        
                is_low_temperature_bool = np.where(temperature < temperature_threshold,True,False)

                offwrist_estimate = np.where(np.logical_and(is_low_temperature_bool,is_normalized_activity_median_low_bool),0,1)
                # plt.plot(datetime_stamps,offwrist_estimate,label="offwrist_estimate")

                print("\nstarting recursive refinement verbose")
                new_refiner = self.__class__(**self.input_parameters)
                refined_day_offwrist_estimate = new_refiner.refine(offwrist_estimate,activity,activity_median,temperature,normalized_temperature_variance,temperature_derivative,temperature_derivative_variance,temperature_threshold,ashman_d,is_normalized_activity_median_low_int,is_low_temperature_bool,self.filter_half_window_length,self.is_lumus_file,cropped_additional_information,epoch_hour=self.epoch_hour,do_near_all_off_detection=False,verbose=3,datetime_stamps=datetime_stamps)
                print("ending recursive refinement verbose\n")
                # plt.plot(datetime_stamps,refined_day_offwrist_estimate,label="refined_estimate")

                # plt.legend()
                # plt.grid()                
                # plt.show()

                refined_offwrist[region_start:region_end] = refined_day_offwrist_estimate
                del new_refiner
        
        return refined_offwrist
        
    
    def near_all_off_detection(self,):
        self.minimum_awake_median_activity_level = 2000

        is_activity_median_high_bool = np.where(self.activity_median > self.minimum_awake_median_activity_level,True,False)
        is_high_temperature_bool = np.logical_not(self.is_low_temperature_bool)
        onwrist_estimate = np.where(np.logical_and(is_high_temperature_bool,is_activity_median_high_bool),0,1)

        onwrist_periods = zero_sequences(onwrist_estimate)
        onwrist_periods_df = self.periods_to_df(onwrist_periods)
        onwrist_periods_df["length"] = onwrist_periods_df["end"] - onwrist_periods_df["start"]

        if self.verbose > 1:
                print("initial near_all_off_detection onwrist_periods")
                if self.datetime_stamps_available:
                        print(self.add_datetime_stamps(onwrist_periods_df))

        onwrist_count = len(onwrist_periods)
        if onwrist_count > 1:
                next_onwrist = 1
                while (next_onwrist < onwrist_count):
                        # print(onwrist_periods)

                        middle_offwrist_start = onwrist_periods[next_onwrist-1,1]
                        middle_offwrist_end = onwrist_periods[next_onwrist,0]
                        middle_offwrist_length = middle_offwrist_end - middle_offwrist_start

                        if middle_offwrist_length < self.minimum_offwrist_length:
                                onwrist_periods[next_onwrist,0] = onwrist_periods[next_onwrist-1,0]
                                onwrist_periods = np.delete(onwrist_periods,next_onwrist-1,axis=0)
                                onwrist_count = len(onwrist_periods)

                        else:
                                maximum_offwrist_length = onwrist_periods_df.at[next_onwrist,"length"] + onwrist_periods_df.at[next_onwrist-1,"length"]
                                if middle_offwrist_length < maximum_offwrist_length:
                                        middle_offwrist_temperature = self.temperature[middle_offwrist_start:middle_offwrist_end]
                                        low_temperature_proportion = below_prop(middle_offwrist_temperature,self.temperature_threshold)
                                        if low_temperature_proportion <= 0.05:
                                                onwrist_periods[next_onwrist,0] = onwrist_periods[next_onwrist-1,0]
                                                onwrist_periods = np.delete(onwrist_periods,next_onwrist-1,axis=0)
                                                onwrist_count = len(onwrist_periods)
                                        else:
                                                next_onwrist += 1

                                else:
                                        next_onwrist += 1

        onwrist_periods_df = self.periods_to_df(onwrist_periods)
        onwrist_periods_df["length"] = onwrist_periods_df["end"] - onwrist_periods_df["start"]

        onwrist_periods_df = onwrist_periods_df[onwrist_periods_df["length"] > self.minimum_onwrist_length]
        onwrist_periods_df.index = np.arange(len(onwrist_periods_df))
        onwrist_periods = self.df_to_periods(onwrist_periods_df)

        if self.verbose > 1:
                print("filtered near_all_off_detection onwrist_periods")
                if self.datetime_stamps_available:
                        print(self.add_datetime_stamps(onwrist_periods_df))

        refined_offwrist = self.onwrist_border_refinement(onwrist_periods_df)
        onwrist_periods = zero_sequences(1-refined_offwrist)

        if len(onwrist_periods) > 0:
                onwrist = np.zeros(self.data_length)
                for period in onwrist_periods:
                        onwrist[period[0]:period[1]] = 1.0
                
                self.offwrist_periods = zero_sequences(onwrist)
                self.offwrist_periods = np.array(self.offwrist_periods)
                self.offwrist_periods_df = self.periods_to_df(self.offwrist_periods)

                # self.refined_offwrist_periods = []
                if self.verbose > 1:
                        print("filtered near_all_off_detection offwrist_periods_df")
                        if self.datetime_stamps_available:
                                print(self.add_datetime_stamps(self.offwrist_periods_df))

                # self.second_stage_refinement()
                # self.onwrist_refinement(onwrist_periods)
                self.refined_offwrist_periods = self.offwrist_periods
                self.is_bimodal = True
                
        else:
                self.refined_offwrist_periods = [[0,self.data_length]]


    def description_report_based_filter(self):
        """Complex filter based on a series of features computed with the most
        refined offwrist periods.
        """
        
        self.report = describe_offwrist_periods(self.refined_offwrist_periods,self.activity,self.temperature,self.normalized_temperature_variance,self.datetime_stamps,self.temperature_threshold,self.activity_threshold,self.sleep_low_activity_threshold,additional_information=self.additional_information,is_lumus_file=self.is_lumus_file,segments=7,verbose=False)
        self.sleep_filtered_report = describe_offwrist_periods(self.sleep_filtered_offwrist_periods,self.activity,self.temperature,self.normalized_temperature_variance,self.datetime_stamps,self.temperature_threshold,self.activity_threshold,self.sleep_low_activity_threshold,additional_information=self.additional_information,is_lumus_file=self.is_lumus_file,segments=7,verbose=False)
        
        num_report = len(self.report)

        if self.verbose:
                print("full self.valley_peak_offwrists_df")
                print(self.valley_peak_offwrists_df)
                print("self.valley_peak_offwrists_df report")

        full_valley_peak_offwrist_periods = self.df_to_periods(self.valley_peak_offwrists_df)
        self.valley_peak_report = describe_offwrist_periods(full_valley_peak_offwrist_periods,self.activity,self.temperature,self.normalized_temperature_variance,self.datetime_stamps,self.temperature_threshold,self.activity_threshold,self.sleep_low_activity_threshold,additional_information=self.additional_information,is_lumus_file=self.is_lumus_file,true_offwrist=self.true_offwrist,segments=7,verbose=False)
        if self.verbose:
                print(self.valley_peak_report)


        # Defines validity thresholds using quantile levels of the user-inputted 
        # additional information
        quantiles = [0.05,0.1,0.25,0.5]
        if self.verbose:
                print("quantiles",quantiles)
        if not self.is_lumus_file:
            temperature_difference = self.additional_information["dif"]
            temperature_difference_variance = var_filter(temperature_difference,3)

            temperature_difference_quantiles = np.quantile(temperature_difference[self.short_offwrist_only_mask],quantiles)
            temperature_difference_variance_quantiles = np.quantile(temperature_difference_variance[self.short_offwrist_only_mask],quantiles)
            if self.verbose:
                print("temperature_difference_quantiles: ",temperature_difference_quantiles)
                print("temperature_difference_variance_quantiles: ",temperature_difference_variance_quantiles)

            temperature_difference_minimum = temperature_difference_quantiles[1]
            if temperature_difference_minimum < 0.2:
                temperature_difference_minimum = 0.2
        else:
            capacitive_difference = self.additional_information["capsensor_dif"]
            capacitive_difference_quantiles = np.quantile(capacitive_difference[self.short_offwrist_only_mask],quantiles)
            if self.verbose:
                print("capacitive_difference_quantiles: ",capacitive_difference_quantiles)
            capacitive_difference_minimum = capacitive_difference_quantiles[1]

        if self.verbose:
                print("self.report\n",self.report)
                print("self.sleep_filtered_report\n",self.sleep_filtered_report)
                print("self.refined_offwrist_periods_df\n",self.refined_offwrist_periods_df)

        self.report["valley_peak"] = self.refined_offwrist_periods_df["valley_peak"].values
        # self.sleep_filtered_report["valley_peak"] = self.sleep_filtered_offwrist_periods_df["valley_peak"].values
        self.report["filtered"] = "NO"
        if self.verbose:
                print("self.report\n",self.report)

                print("self.refined_offwrist_periods\n",self.refined_offwrist_periods)

        # input()

        # self.offwrist_minimum_temperature_difference_median = 0.75

        # Offwrist periods are sequentially evaluated based on their descriptive
        # statistical features
        valid_offwrist_mask = []
        self.description_filter_deleted_periods = np.array([])
        if (self.refined_offwrist_periods.shape[0] > 0):
            for o in range(num_report):
                is_valid = True
                # Only short offwrist periods can be filtered out
                if (self.report.at[o,"length"] < self.long_offwrist_length) and (not self.is_highly_separable):
                        if is_valid and (self.do_report_zero_activity_proportion_filter and (self.report.at[o,"activity_zero_proportion"] < self.report_zero_activity_proportion_minimum)):
                            # if is_valid and (self.do_report_zero_activity_proportion_filter and (self.report.at[o,"activity_zero_proportion"] < self.report_zero_activity_proportion_minimum)):
                                if self.do_report_low_activity_after:
                                        if is_valid and (self.do_report_low_activity_proportion_filter and (self.report.at[o,"low_activity_proportion"] < self.report_low_activity_proportion_minimum)):
                                                is_valid = False
                                                self.report.at[o,"filtered"] = "zero_activity_filter"

                        if (self.skip_description_filters or (not ( self.report.at[o,"valley_peak"] and (self.report.at[o,"length"] <= self.short_offwrist_length) ))):
                        # if True:
                            
                            # If any of this invalidation criteria are met, the offwrist
                            # will be filtered out. All numerical parameters used here we-
                            # obtained empirically using historical data.

                            if self.do_report_border_concentration_filter and ((self.report.at[o,"border_activity_concentration"] < 0.35) and (self.report.at[o,"border_temperature_concentration"] < 0.35)):
                            # if self.do_report_border_concentration_filter and ((self.report.at[o,"border_activity_concentration"] < 0.35) and (self.report.at[o,"border_temperature_concentration"] < 0.35)):
                                is_valid = False
                                self.report.at[o,"filtered"] = "border_concentration_filter"

                            if is_valid and (self.do_report_activity_around_filter and ((self.report.at[o,"high_activity_proportion_after"] < 0.2) and (self.report.at[o,"high_activity_proportion_before"] < 0.2))):
                            # if is_valid and (self.do_report_activity_around_filter and ((self.report.at[o,"high_activity_proportion_after"] < 0.2) and (self.report.at[o,"high_activity_proportion_before"] < 0.2))):
                                is_valid = False
                                self.report.at[o,"filtered"] = "low_activity_filter"

                            if not self.do_report_low_activity_after:
                                if is_valid and (self.do_report_low_activity_proportion_filter and (self.report.at[o,"low_activity_proportion"] < self.report_low_activity_proportion_minimum)):
                                        is_valid = False
                                        self.report.at[o,"filtered"] = "low_activity_filter"

                            if is_valid and (self.do_report_border_activity_filter and (((self.report.at[o,"high_activity_proportion_after"] < self.report_activity_around_minimum) or (self.report.at[o,"high_activity_proportion_before"] < self.report_activity_around_minimum)) and ((self.report.at[o,"border_activity_concentration"] < self.border_concentratition_minimum) and (self.report.at[o,"border_temperature_concentration"] < self.border_concentratition_minimum)))):
                            # if is_valid and (self.do_report_border_activity_filter and (((self.report.at[o,"high_activity_proportion_after"] < self.report_activity_around_minimum) or (self.report.at[o,"high_activity_proportion_before"] < self.report_activity_around_minimum)) and ((self.report.at[o,"border_activity_concentration"] < self.border_concentratition_minimum) and (self.report.at[o,"border_temperature_concentration"] < self.border_concentratition_minimum)))):
                                is_valid = False
                                self.report.at[o,"filtered"] = "border_activity_filter"

                            # if is_valid and (True and (self.report.at[o,"low_temperature_proportion"] < 0.6)):
                            #     is_valid = False

                            if (is_valid and (not self.is_lumus_file)):
                                if (self.do_temperature_difference_filter and (self.report.at[o,"temperature_difference_median"] > self.offwrist_minimum_temperature_difference_median)):
                                        # if (not self.refined_offwrist_periods_df.at[o,"valley_peak"]):
                                        is_valid = False
                                        self.report.at[o,"filtered"] = "temperature_difference_median"


                            # Filtered out offwrist periods may be spared
                            # if they meet addtional criteria
                            # if self.do_statistical_measure_filter and ~is_valid:
                            #     if self.is_lumus_file:
                            #         if self.report.at[o,"capacitive_difference_variance"] < capacitive_difference_minimum:
                            #             is_valid = True
                            #     else:
                            #         if self.report.at[o,"temperature_difference_median"] < temperature_difference_minimum:
                            #             is_valid = True


                valid_offwrist_mask.append(is_valid)

            valid_offwrist_mask = np.array(valid_offwrist_mask)

            self.description_filter_deleted_periods = 1-valid_offwrist_mask.astype(int)
            self.description_filter_deleted_periods = self.description_filter_deleted_periods.nonzero()[0]

            self.filtered_report = self.report.loc[valid_offwrist_mask,:]
            self.filtered_report.index = np.arange(len(self.filtered_report))
        else:
            self.filtered_report = self.report.copy()


        if self.is_bimodal and self.do_description_report_based_filter:
            if len(valid_offwrist_mask) > 0:
                self.description_filtered_offwrist_periods = self.refined_offwrist_periods[valid_offwrist_mask].copy()
            else:
                self.description_filtered_offwrist_periods = np.array([])

            if self.verbose:
                print("self.description_filtered_offwrist_periods")
                print(self.description_filtered_offwrist_periods)
            
            self.refined_offwrist_periods = self.description_filtered_offwrist_periods.copy()

  
    def surrounded_onwrist_filter(self,):
        """After the detections are reasonably defined, the aim is to eliminate
        invalid periods. This functions filters onwrist periods that are surroun-
        ded by relatively long offwrist periods for they are more likely a part
        of a single offwrist periods where the device was exposed to movement.
        """

        # self.refined_offwrist_periods = np.array(self.refined_offwrist_periods)
        # self.refined_offwrist_periods_df = self.periods_to_df(self.refined_offwrist_periods)
        
        self.onwrist_periods = zero_sequences(1-self.refined_offwrist)
        self.onwrist_periods_df = self.describe_onwrist_periods()
        
        onwrist_count = len(self.onwrist_periods_df)

        minimum_surrounding_offwrist_length = 80
        if self.verbose:
             print("surrounded_onwrist_filter")
             print("minimum_surrounding_offwrist_length",minimum_surrounding_offwrist_length)
             print("minimum_preceding_onwrist_period_length",self.minimum_preceding_onwrist_period_length)
             print("self.onwrist_periods")
             print(self.onwrist_periods_df)
        #      print(self.onwrist_periods)

        if len(self.refined_offwrist_periods) > 0:
                valid_onwrist_indexes = []
                for i in range(onwrist_count):
                        valid_onwrist = True
                        onwrist_length = self.onwrist_periods_df.at[i,"length"]
                        preceding_offwrist_length = 0
                        suceeding_offwrist_length = 0

                        onwrist_temperature = self.temperature[self.onwrist_periods_df.at[i,"start"]:self.onwrist_periods_df.at[i,"end"]]
                        onwrist_low_temperature_proportion = below_prop(onwrist_temperature,self.temperature_threshold)

                        # if onwrist_length <= self.minimum_preceding_onwrist_period_length:
                        if i == 0:
                                if onwrist_count > 1:
                                        suceeding_offwrist_length = self.onwrist_periods_df.at[1,"start"] - self.onwrist_periods_df.at[0,"end"]
                                else:
                                        suceeding_offwrist_length = self.data_length - self.onwrist_periods_df.at[0,"end"]
                                        
                                if self.onwrist_periods_df.at[0,"start"] == 0:
                                        if (suceeding_offwrist_length > minimum_surrounding_offwrist_length):
                                                if onwrist_length <= self.minimum_preceding_onwrist_period_length:
                                                        valid_onwrist = False
                                                else:
                                                        if ((onwrist_length < suceeding_offwrist_length) and (onwrist_low_temperature_proportion > 0.75)):
                                                                valid_onwrist = False
                                                        
                                else:
                                        preceding_offwrist_length = self.onwrist_periods_df.at[0,"start"]
                                        if (preceding_offwrist_length > minimum_surrounding_offwrist_length) and (suceeding_offwrist_length > minimum_surrounding_offwrist_length):
                                                if onwrist_length <= self.minimum_preceding_onwrist_period_length:
                                                        valid_onwrist = False
                                                else:
                                                        if ((onwrist_length < suceeding_offwrist_length+preceding_offwrist_length) and (onwrist_low_temperature_proportion > 0.75)):
                                                                valid_onwrist = False
                                
                        else:
                                preceding_offwrist_length = self.onwrist_periods_df.at[i,"start"] - self.onwrist_periods_df.at[i-1,"end"]
                                if i == onwrist_count-1:
                                        if self.onwrist_periods_df.at[i,"end"] == self.data_length:
                                                if (preceding_offwrist_length > minimum_surrounding_offwrist_length):
                                                        if onwrist_length <= self.minimum_preceding_onwrist_period_length:
                                                                valid_onwrist = False
                                                        else:
                                                                if ((onwrist_length < preceding_offwrist_length) and (onwrist_low_temperature_proportion > 0.75)):
                                                                        valid_onwrist = False
                                        else:
                                                suceeding_offwrist_length = self.data_length - self.onwrist_periods_df.at[i,"end"]
                                                if (preceding_offwrist_length > minimum_surrounding_offwrist_length) and (suceeding_offwrist_length > minimum_surrounding_offwrist_length):
                                                        if onwrist_length <= self.minimum_preceding_onwrist_period_length:
                                                                valid_onwrist = False
                                                        else:   
                                                                if ((onwrist_length < suceeding_offwrist_length+preceding_offwrist_length) and (onwrist_low_temperature_proportion > 0.75)):
                                                                        valid_onwrist = False
                                else:
                                        suceeding_offwrist_length = self.onwrist_periods_df.at[i+1,"start"] - self.onwrist_periods_df.at[i,"end"]
                                        if (preceding_offwrist_length > minimum_surrounding_offwrist_length) and (suceeding_offwrist_length > minimum_surrounding_offwrist_length):
                                                if onwrist_length <= self.minimum_preceding_onwrist_period_length:
                                                        valid_onwrist = False
                                                else:
                                                        if ((onwrist_length < suceeding_offwrist_length+preceding_offwrist_length) and (onwrist_low_temperature_proportion > 0.75)):
                                                                valid_onwrist = False

                        if self.verbose:
                                print("current onwrist")
                                print(self.onwrist_periods_df.loc[i,:])
                                print("preceding_offwrist_length", preceding_offwrist_length)
                                print("suceeding_offwrist_length", suceeding_offwrist_length)
                                print("onwrist_low_temperature_proportion", onwrist_low_temperature_proportion)
                                print("valid_onwrist", valid_onwrist)

                        if valid_onwrist:
                                valid_onwrist_indexes.append(i)

                if len(valid_onwrist_indexes) != onwrist_count:
                        self.refined_onwrist_periods_df = self.onwrist_periods_df.loc[valid_onwrist_indexes,:]
                        self.refined_onwrist_periods = self.df_to_periods(self.refined_onwrist_periods_df)

                        self.refined_offwrist = np.zeros(self.data_length)
                        for onwrist_index in self.refined_onwrist_periods:
                                self.refined_offwrist[onwrist_index[0]:onwrist_index[1]] = 1.0

                        self.refined_offwrist_periods = np.array(zero_sequences(self.refined_offwrist,ending_at_n=True))
                

    def third_stage_refinement(self,):
        """Applies a sequence of complex filters to the refined offwrist pe-
        riods. Starting with feature-based filters, a sleep estimate based 
        filters and a final feature-based filter.
        """

        # Consolidating periods from second stage refinement before computing
        # features for filtering
        self.refined_offwrist_periods = np.array(self.refined_offwrist_periods)
        self.refined_offwrist_periods_df = self.periods_to_df(self.refined_offwrist_periods)
        self.refined_offwrist_periods_df["length"] = self.refined_offwrist_periods_df["end"] - self.refined_offwrist_periods_df["start"]

        self.half_day_length = 12*self.epoch_hour

        self.refined_offwrist_periods_df["low_temperature_proportion"] = np.array([below_prop(self.temperature[o[0]:o[1]],self.temperature_threshold) for o in self.refined_offwrist_periods])
        self.low_temperature_filter()

        self.refine_temperature_threshold()
        if self.do_temperature_threshold_refinement:
                self.refined_offwrist_periods_df["low_temperature_proportion"] = np.array([below_prop(self.temperature[o[0]:o[1]],self.temperature_threshold) for o in self.refined_offwrist_periods])
                self.low_temperature_filter()

        self.refined_offwrist_periods_df["low_activity_proportion"] = np.array([below_prop(self.activity[o[0]:o[1]],self.activity_threshold) for o in self.refined_offwrist_periods])
        self.refined_offwrist_periods_df["zero_activity_proportion"] = np.array([zero_prop(self.activity[o[0]:o[1]]) for o in self.refined_offwrist_periods])
        self.low_activity_filter()

        self.low_temperature_variance_filter()

        self.offwrist_length_filter()

        # self.offwrist_maximum_temperature_difference_median = 1.0
        # # self.offwrist_maximum_capsensor1_std = 6.0
        # self.offwrist_minimum_capsensor1_median = 165
        # self.offwrist_minimum_capsensor2_median = 120

        if self.offwrists_need_decreasing_temperature:
                self.refined_offwrist_periods_df = self.compute_temperature_variations(self.refined_offwrist_periods_df)
                self.refined_offwrist_periods_df = self.refined_offwrist_periods_df[self.refined_offwrist_periods_df["decrease_ratio"] > 0]
                self.refined_offwrist_periods_df.index = np.arange(self.refined_offwrist_periods_df.shape[0])

                if self.verbose > 1:
                        print("decreasing temperature filtered refined offwrist_periods\n",self.refined_offwrist_periods)
                        if self.datetime_stamps_available:
                                print(self.add_datetime_stamps(self.refined_offwrist_periods_df))
                        
                self.refined_offwrist_periods = self.df_to_periods(self.refined_offwrist_periods_df)

        if not self.is_lumus_file:
                self.refined_offwrist_periods_df["temperature_difference_median"] = np.array([np.median(self.additional_information["dif"][o[0]:o[1]]) for o in self.refined_offwrist_periods])
                if self.verbose > 1:
                        print("length filtered refined offwrist_periods\n",self.refined_offwrist_periods)
                        if self.datetime_stamps_available:
                                print(self.add_datetime_stamps(self.refined_offwrist_periods_df))

                self.refined_offwrist_periods_df = self.refined_offwrist_periods_df[self.refined_offwrist_periods_df["temperature_difference_median"] < self.offwrist_maximum_temperature_difference_median]
                self.refined_offwrist_periods_df.index = np.arange(self.refined_offwrist_periods_df.shape[0])

                if self.verbose > 1:
                        print("temperature_difference_median filtered refined offwrist_periods\n",self.refined_offwrist_periods)
                        if self.datetime_stamps_available:
                                print(self.add_datetime_stamps(self.refined_offwrist_periods_df))
                        
                self.refined_offwrist_periods = self.df_to_periods(self.refined_offwrist_periods_df)

        else:
                self.refined_offwrist_periods_df["capsensor1_std"] = np.array([np.std(self.additional_information["c1"][o[0]:o[1]]) for o in self.refined_offwrist_periods])
                self.refined_offwrist_periods_df["capsensor2_std"] = np.array([np.std(self.additional_information["c2"][o[0]:o[1]]) for o in self.refined_offwrist_periods])
                self.refined_offwrist_periods_df["capsensor1_median"] = np.array([np.median(self.additional_information["c1"][o[0]:o[1]]) for o in self.refined_offwrist_periods])
                self.refined_offwrist_periods_df["capsensor2_median"] = np.array([np.median(self.additional_information["c2"][o[0]:o[1]]) for o in self.refined_offwrist_periods])
                if self.verbose > 1:
                        print("length filtered refined offwrist_periods\n",self.refined_offwrist_periods)
                        if self.datetime_stamps_available:
                                print(self.add_datetime_stamps(self.refined_offwrist_periods_df))

                
                valid_indexes = []
                for offwrist in self.refined_offwrist_periods_df.index:
                        if self.refined_offwrist_periods_df.at[offwrist,"length"] >= self.half_day_length:
                                valid_indexes.append(offwrist)
                        else:
                                if ( 
                                     ( (self.refined_offwrist_periods_df.at[offwrist,"capsensor1_std"] < self.offwrist_maximum_capsensor1_std) and (self.refined_offwrist_periods_df.at[offwrist,"capsensor2_std"] < self.offwrist_maximum_capsensor2_std) ) and
                                     ( (self.refined_offwrist_periods_df.at[offwrist,"capsensor1_median"] > self.offwrist_minimum_capsensor1_median) and (self.refined_offwrist_periods_df.at[offwrist,"capsensor2_median"] > self.offwrist_minimum_capsensor2_median) )
                                ):
                                        valid_indexes.append(offwrist)

                self.refined_offwrist_periods_df = self.refined_offwrist_periods_df.loc[valid_indexes,:]
                self.refined_offwrist_periods_df.index = np.arange(self.refined_offwrist_periods_df.shape[0])

                if self.verbose > 1:
                        print("offwrist_minimum_capsensor1_median filtered refined offwrist_periods\n",self.refined_offwrist_periods)
                        if self.datetime_stamps_available:
                                print(self.add_datetime_stamps(self.refined_offwrist_periods_df))
                        
                self.refined_offwrist_periods = self.df_to_periods(self.refined_offwrist_periods_df)

        # Consolidating periods after initial filters and computing features 
        # for subsequent comparisons
        self.after_initial_refinement_offwrist_periods = self.refined_offwrist_periods.copy()
        self.after_initial_refinement_offwrist_report = describe_offwrist_periods(self.refined_offwrist_periods,self.activity,self.temperature,self.normalized_temperature_variance,self.datetime_stamps,self.temperature_threshold,self.activity_threshold,self.activity_threshold,additional_information=self.additional_information,is_lumus_file=self.is_lumus_file,segments=7,verbose=False)
        if self.verbose > 1:
            print("after_initial_refinement_offwrist offwrist_periods")
            print(self.refined_offwrist_periods)
            if self.datetime_stamps_available:
                print(self.add_datetime_stamps(self.refined_offwrist_periods_df))
        
        self.after_initial_refinement_offwrist = np.ones(self.data_length)
        for offwrist_index in self.refined_offwrist_periods:
            self.after_initial_refinement_offwrist[offwrist_index[0]:offwrist_index[1]] = 0.0


        # Offwrist periods longer than 4 hours are considered highly stable and
        # will not be used in the sleep estimation
        self.short_offwrist_only_mask = np.full(self.data_length,True,dtype=bool)
        for o in range(self.refined_offwrist_periods.shape[0]):
            if self.refined_offwrist_periods_df["length"].iat[o] >= self.long_offwrist_length:
                self.short_offwrist_only_mask[self.refined_offwrist_periods[o][0]:self.refined_offwrist_periods[o][1]] = False
        self.valid_activity = self.activity[self.short_offwrist_only_mask].copy()
        self.valid_temperature = self.temperature[self.short_offwrist_only_mask].copy()

        self.estimate_sleep_then_filter()
        if self.do_sleep_filter:
            self.refined_offwrist_periods = self.sleep_filtered_offwrist_periods
        
        if self.verbose:
                print(self.refined_offwrist_periods)

        self.sleep_filtered_offwrist = np.ones(self.data_length)
        for offwrist_index in self.refined_offwrist_periods:
            self.sleep_filtered_offwrist[offwrist_index[0]:offwrist_index[1]] = 0.0

        self.analyze_sleep_borders()

        self.valley_peak_offwrist_algorithm(include_static=True,static_quantile=0.95)

        self.surrounded_valley_peak_detection()

        self.low_temperature_proportion = below_prop(self.temperature,self.temperature_threshold)

        if self.verbose:
                print(self.short_sleep_border_offwrist)

        if self.do_forbidden_zone:
                allowed_sleep_filtered_offwrist_periods = []
                for offwrist_index in self.refined_offwrist_periods:
                    forbidden = self.forbidden_zone[offwrist_index[0]:offwrist_index[1]]
                    if np.sum(forbidden) == 0:
                        allowed_sleep_filtered_offwrist_periods.append(offwrist_index)

                offwrists_count = len(self.refined_offwrist_periods)
                self.sleep_filtered_offwrist_periods = allowed_sleep_filtered_offwrist_periods

                if len(self.sleep_filtered_offwrist_periods) < offwrists_count:
                        self.forbidden_filtered = True

                self.refined_offwrist_periods = self.sleep_filtered_offwrist_periods

                self.sleep_filtered_offwrist = np.ones(self.data_length)
                for offwrist_index in self.sleep_filtered_offwrist_periods:
                    self.sleep_filtered_offwrist[offwrist_index[0]:offwrist_index[1]] = 0.0


        sleep_filtered_offwrist_periods_df = self.periods_to_df(self.sleep_filtered_offwrist_periods)
        sleep_filtered_offwrist_periods_df = self.add_datetime_stamps(sleep_filtered_offwrist_periods_df)
        sleep_filtered_offwrist_periods_df["sleep_border"] = False
        sleep_filtered_offwrist_periods_df["length"] = sleep_filtered_offwrist_periods_df["end"] - sleep_filtered_offwrist_periods_df["start"]

        self.sleep_filtered_offwrist_periods_df = sleep_filtered_offwrist_periods_df
        
        self.is_bimodal, self.is_low_activity = self.check_bimodality()
        if self.is_bimodal:
                if self.verbose:
                        print("bimodal")

                computed_short_valley_peak_offwrists_df = self.computed_valley_peak_offwrists_df[self.computed_valley_peak_offwrists_df["length"] < self.long_offwrist_length]
                if self.do_valley_peak_algorithm:
                        short_valley_peak_offwrists_df = self.valley_peak_offwrists_df[self.valley_peak_offwrists_df["length"] < self.long_offwrist_length]
                else:
                        short_valley_peak_offwrists_df = self.periods_to_df([])

                short_sleep_filtered_offwrists_df = self.sleep_filtered_offwrist_periods_df[self.sleep_filtered_offwrist_periods_df["length"] < self.long_offwrist_length]
                long_sleep_filtered_offwrists_df = self.sleep_filtered_offwrist_periods_df[self.sleep_filtered_offwrist_periods_df["length"] >= self.long_offwrist_length]

                if self.verbose:
                        print("short_valley_peak_offwrists_df\n",short_valley_peak_offwrists_df)
                        print("long_sleep_filtered_offwrists_df\n",long_sleep_filtered_offwrists_df)
                        print("short_sleep_filtered_offwrists_df\n",short_sleep_filtered_offwrists_df)

                short_sleep_filtered_offwrists_df["valley_peak_agreement"] = [1-zero_prop(self.computed_valley_peak_offwrist[short_sleep_filtered_offwrists_df.at[offwrist,"start"]:short_sleep_filtered_offwrists_df.at[offwrist,"end"]]) for offwrist in short_sleep_filtered_offwrists_df.index]
                short_sleep_filtered_offwrists_df["offwrist_agreement"] = [zero_prop(self.true_offwrist[short_sleep_filtered_offwrists_df.at[offwrist,"start"]:short_sleep_filtered_offwrists_df.at[offwrist,"end"]]) for offwrist in short_sleep_filtered_offwrists_df.index]

                if self.verbose:
                        print("short_sleep_filtered_offwrists_df\n",short_sleep_filtered_offwrists_df)
                if self.do_valley_peak_filter:
                        short_sleep_filtered_offwrists_df = short_sleep_filtered_offwrists_df[short_sleep_filtered_offwrists_df["valley_peak_agreement"] > 0.75]
                
                if self.verbose:
                        print("filtered short_sleep_filtered_offwrists_df\n",short_sleep_filtered_offwrists_df)


                short_valley_peak_offwrists_df["offwrist_agreement"] = [zero_prop(self.true_offwrist[short_valley_peak_offwrists_df.at[offwrist,"start"]:short_valley_peak_offwrists_df.at[offwrist,"end"]]) for offwrist in short_valley_peak_offwrists_df.index]
                
                # positive_temperature_derivative_indexes

                # median_temperature_derivative_variance = median_filter(self.temperature_derivative_variance,feature_extraction_filters_half_window_length)
                # temperature_derivative_variance_scaled = (1/np.max(np.absolute(temperature_derivative_variance)))*temperature_derivative_variance

                long_sleep_filtered_offwrists_df["valley_peak"] = False
                short_sleep_filtered_offwrists_df["valley_peak"] = False
                short_valley_peak_offwrists_df["valley_peak"] = True
                self.refined_offwrist_periods_df = pd.concat([long_sleep_filtered_offwrists_df,short_sleep_filtered_offwrists_df,short_valley_peak_offwrists_df],ignore_index=True)

                if self.verbose:
                        print("sleep_border_offwrists_df\n",self.sleep_border_offwrists_df)

                # self.refined_offwrist_periods_df = self.valley_peak_offwrists_df

                self.refined_offwrist_periods_df.sort_values(by="start",inplace=True)
                
                self.refined_offwrist_periods = self.df_to_periods(self.refined_offwrist_periods_df).astype(int)

                self.refined_offwrist_periods_df["low_temperature_proportion"] = np.array([below_prop(self.temperature[o[0]:o[1]],self.temperature_threshold) for o in self.refined_offwrist_periods])
                self.refined_offwrist_periods_df["low_activity_proportion"] = np.array([below_prop(self.activity[o[0]:o[1]],self.activity_threshold) for o in self.refined_offwrist_periods])
                self.refined_offwrist_periods_df["zero_activity_proportion"] = np.array([zero_prop(self.activity[o[0]:o[1]]) for o in self.refined_offwrist_periods])
                self.refined_offwrist_periods_df["length"] = self.refined_offwrist_periods_df["end"] - self.refined_offwrist_periods_df["start"]
                # self.refined_offwrist_periods_df["sleep_border"] = False
                self.refined_offwrist_periods_df["valley_peak_agreement"] = [1-zero_prop(self.valley_peak_offwrist[o[0]:o[1]]) for o in self.refined_offwrist_periods]
                self.refined_offwrist_periods_df["offwrist_agreement"] = [zero_prop(self.true_offwrist[o[0]:o[1]]) for o in self.refined_offwrist_periods]
                self.sleep_filtered_offwrist_periods = self.refined_offwrist_periods.copy()

                if self.verbose:
                        print("sleep_border_offwrists_df\n",self.sleep_border_offwrists_df)
                        print("sleep_filtered_offwrist_periods_df\n",self.sleep_filtered_offwrist_periods_df)
                        print("refined_offwrist_periods_df\n",self.refined_offwrist_periods_df)
                        print("refined_offwrist_periods\n",self.refined_offwrist_periods)

        else:
                if self.verbose:
                        print("unimodal")
        
                if self.is_low_activity or (self.is_lumus_file and (self.activity_median_high_proportion < self.lumus_file_minimum_activity_median_high_proportion)):
                        if self.verbose:
                                print("all off")

                        if self.do_near_all_off_detection:
                                self.near_all_off_detection()

                        refined_offwrist_periods_df = self.periods_to_df(self.refined_offwrist_periods)
                        refined_offwrist_periods_df = self.add_datetime_stamps(refined_offwrist_periods_df)
                        refined_offwrist_periods_df["valley_peak"] = False

                        self.refined_offwrist_periods_df = refined_offwrist_periods_df
                        
                else:
                        if self.verbose:
                                print("all on")

                        if np.sum(self.short_sleep_border_offwrist) > 0:
                                if self.verbose:
                                        print("changed to bimodal")
        
                                self.refined_offwrist_periods = self.valley_peak_offwrists
                                refined_offwrist_periods_df = self.periods_to_df(self.refined_offwrist_periods)
                                refined_offwrist_periods_df = self.add_datetime_stamps(refined_offwrist_periods_df)
                                refined_offwrist_periods_df["valley_peak"] = True

                                self.refined_offwrist_periods_df = refined_offwrist_periods_df
                                
                        else:
                                self.refined_offwrist_periods = []
                                refined_offwrist_periods_df = self.periods_to_df(self.refined_offwrist_periods)
                                refined_offwrist_periods_df = self.add_datetime_stamps(refined_offwrist_periods_df)
                                refined_offwrist_periods_df["valley_peak"] = False

                                self.refined_offwrist_periods_df = refined_offwrist_periods_df


        self.refined_offwrist_periods = np.array(self.refined_offwrist_periods)

        self.refined_offwrist = np.ones(self.data_length)
        for offwrist_index in self.refined_offwrist_periods:
            self.refined_offwrist[offwrist_index[0]:offwrist_index[1]] = 0.0

        if self.verbose:
                print("self.refined_offwrist_periods\n",self.refined_offwrist_periods)
                print("activity_threshold",self.activity_threshold)
                print("sleep_low_activity_threshold",self.sleep_low_activity_threshold)
                print("refined_low_activity_threshold",self.refined_low_activity_threshold)
        self.description_report_based_filter()


        # Final third stage refined offwrist data
        # if self.do_forbidden_zone:
        #         self.refined_offwrist = np.ones(self.data_length)
        #         for offwrist_index in self.refined_offwrist_periods:
        #             forbidden = self.forbidden_zone[offwrist_index[0]:offwrist_index[1]]
        #             if np.sum(forbidden) == 0:
        #                 self.refined_offwrist[offwrist_index[0]:offwrist_index[1]] = 0.0
        # else:
        self.refined_offwrist = np.ones(self.data_length)
        for offwrist_index in self.refined_offwrist_periods:
            self.refined_offwrist[offwrist_index[0]:offwrist_index[1]] = 0.0
        
        self.refined_offwrist_periods = np.array(zero_sequences(self.refined_offwrist))

        self.do_surrounded_onwrist_filter = False
        self.do_surrounded_onwrist_filter = True
        if self.do_surrounded_onwrist_filter:
                self.surrounded_onwrist_filter()
        
        self.final_border_refinement = False
        self.final_border_refinement = True
        if self.final_border_refinement:
                offwrist_count = len(self.refined_offwrist_periods)
                if offwrist_count > 0:
                        if self.refined_offwrist_periods[0][0] <= self.minimum_offwrist_length:
                                self.refined_offwrist_periods[0][0] = 0
                        
                        if (self.data_length - self.refined_offwrist_periods[offwrist_count-1][1]) <= self.minimum_offwrist_length:
                                self.refined_offwrist_periods[offwrist_count-1][1] = self.data_length
                
                        self.refined_offwrist = np.ones(self.data_length)
                        for offwrist_index in self.refined_offwrist_periods:
                                self.refined_offwrist[offwrist_index[0]:offwrist_index[1]] = 0.0

        if self.verbose:
                print("self.refined_offwrist_periods",self.refined_offwrist_periods)

        if len(self.refined_offwrist_periods) > 0:
                self.refined_offwrist_periods_df = self.periods_to_df(self.refined_offwrist_periods)
                self.refined_offwrist_periods_df["length"] = self.refined_offwrist_periods_df["end"] - self.refined_offwrist_periods_df["start"]
                self.refined_long_offwrist_periods_df = self.refined_offwrist_periods_df[self.refined_offwrist_periods_df["length"] > self.long_offwrist_length]
                self.refined_long_offwrist_periods = self.df_to_periods(self.refined_long_offwrist_periods_df)
        else:
                self.refined_offwrist_periods_df = pd.DataFrame([])
                self.refined_long_offwrist_periods_df = pd.DataFrame([])
                self.refined_long_offwrist_periods = []

        self.refined_long_offwrist = np.ones(self.data_length)
        for offwrist_index in self.refined_long_offwrist_periods:
            self.refined_long_offwrist[offwrist_index[0]:offwrist_index[1]] = 0.0

    def compute_and_print_results_report(self):
        """Compares, if supplied, the actual sleep and offwrist information and
        the refined estimated information. Also computes useful statistics to e-
        valuate the algorithms performance.
        """

        sleep_border_overdetection = 0

        if self.do_sleep_filter and self.is_bimodal and ((len(self.true_offwrist) > 0) and (len(self.true_sleep) > 0)):
            acc,sss,spp,_,_,on_detect,off_detect,over_found,over_found_sleep,long_detection,ofs_location,of_location = scores(self.true_offwrist,self.after_initial_refinement_offwrist,states=self.true_sleep,ofs_locate=True,long_event=self.long_offwrist_length)

            self.after_initial_refinement_offwrist_periods = self.after_initial_refinement_offwrist_periods.tolist()
            self.sleep_filtered_offwrist_periods = self.sleep_filtered_offwrist_periods.tolist()
            self.description_filtered_offwrist_periods = self.description_filtered_offwrist_periods.tolist()
            
            if self.verbose:
                print("self.after_initial_refinement_offwrist_periods\n",self.after_initial_refinement_offwrist_periods)
                print("self.sleep_filtered_offwrist_periods\n",self.sleep_filtered_offwrist_periods)
                print("self.description_filtered_offwrist_periods\n",self.description_filtered_offwrist_periods)

            self.after_initial_refinement_offwrist_report["overdetection"] = False
            self.after_initial_refinement_offwrist_report["sleep_overdetection"] = False
            offwrist_count = len(self.after_initial_refinement_offwrist_report)
            for offwrist_index in range(offwrist_count):
                offwrist_period = self.after_initial_refinement_offwrist_periods[offwrist_index]
                if 1-zero_prop(self.true_offwrist[offwrist_period[0]:offwrist_period[1]]) > 0.4:
                    # If an estimated offwrist period doesn't overlap by at least 
                    # 40% with the true offwrist information, it's considered an
                    # "over" detected period.
                    self.after_initial_refinement_offwrist_report.at[offwrist_index,"overdetection"] = True
                    if 1-zero_prop(self.true_sleep[offwrist_period[0]:offwrist_period[1]]) > 0.4:
                        # If an estimated over detected offwrist period overlaps 
                        # true sleep information by at least 40%, it gets and ex-
                        # tra flag that it is an overdetection during a sleep pe-
                        # riod, the worst case scenario.
                        self.after_initial_refinement_offwrist_report.at[offwrist_index,"sleep_overdetection"] = True

            sleep_filter_deleted_accurate_detections = []
            for d in self.sleep_filter_deleted_periods:
                if not self.after_initial_refinement_offwrist_report.at[d,"overdetection"]:
                    sleep_filter_deleted_accurate_detections.append(d)

            initial_refinement_overdetection_count = len(self.after_initial_refinement_offwrist_report[self.after_initial_refinement_offwrist_report["overdetection"] == True])
            initial_refinement_sleep_overdetection_count = len(self.after_initial_refinement_offwrist_report[self.after_initial_refinement_offwrist_report["sleep_overdetection"] == True])
            initial_refinement_detection_count = offwrist_count - initial_refinement_overdetection_count

            self.sleep_filtered_report["overdetection"] = False
            self.sleep_filtered_report["sleep_overdetection"] = False
            offwrist_count = len(self.sleep_filtered_report)
            for offwrist_index in range(offwrist_count):
                offwrist_period = self.sleep_filtered_offwrist_periods[offwrist_index]
                if 1-zero_prop(self.true_offwrist[offwrist_period[0]:offwrist_period[1]]) > 0.4:
                    self.sleep_filtered_report.at[offwrist_index,"overdetection"] = True
                    if 1-zero_prop(self.true_sleep[offwrist_period[0]:offwrist_period[1]]) > 0.4:
                        self.sleep_filtered_report.at[offwrist_index,"sleep_overdetection"] = True

            if self.verbose:
                print("self.description_filter_deleted_periods\n",self.description_filter_deleted_periods)
                print("self.sleep_filtered_report\n",self.sleep_filtered_report)

            description_filter_deleted_accurate_detections = []
            for d in self.description_filter_deleted_periods:
                if not self.sleep_filtered_report.at[d,"overdetection"]:
                    description_filter_deleted_accurate_detections.append(d)

            sleep_filter_overdetection_count = len(self.sleep_filtered_report[self.sleep_filtered_report["overdetection"] == True])
            sleep_filter_sleep_overdetection_count = len(self.sleep_filtered_report[self.sleep_filtered_report["sleep_overdetection"] == True])
            sleep_filter_detection_count = offwrist_count - sleep_filter_overdetection_count 

            self.delta_overdetection = initial_refinement_overdetection_count - sleep_filter_overdetection_count
            self.delta_sleep_overdetection = initial_refinement_sleep_overdetection_count - sleep_filter_sleep_overdetection_count
            self.delta_detection = initial_refinement_detection_count - sleep_filter_detection_count

            self.filtered_report["overdetection"] = False
            self.filtered_report["sleep_overdetection"] = False
            offwrist_count = len(self.description_filtered_offwrist_periods)
            for offwrist_index in range(offwrist_count):
                offwrist_period = self.description_filtered_offwrist_periods[offwrist_index]
                if 1-zero_prop(self.true_offwrist[offwrist_period[0]:offwrist_period[1]]) > 0.4:
                    self.filtered_report.at[offwrist_index,"overdetection"] = True
                    
                    if self.filtered_report.at[offwrist_index,"valley_peak"]:
                            sleep_border_overdetection += 1

                    if 1-zero_prop(self.true_sleep[offwrist_period[0]:offwrist_period[1]]) > 0.4:
                        self.filtered_report.at[offwrist_index,"sleep_overdetection"] = True

            if self.verbose:
                print("sleep_filter_deleted_periods",self.sleep_filter_deleted_periods)
                print("sleep_filter_deleted_accurate_detections",sleep_filter_deleted_accurate_detections)

                print("after_initial_refinement_offwrist_report")
                print(self.after_initial_refinement_offwrist_report)

                print("description_filter_deleted_periods",self.description_filter_deleted_periods)
                print("description_filter_deleted_accurate_detections",description_filter_deleted_accurate_detections)

                print("self.sleep_filtered_report")
                print(self.sleep_filtered_report)

                print("self.filtered_report")
                print(self.filtered_report)

                print("initial_refinement_overdetection_count", initial_refinement_overdetection_count)
                print("initial_refinement_sleep_overdetection_count", initial_refinement_sleep_overdetection_count)
                print("initial_refinement_detection_count", initial_refinement_detection_count)
                print("sleep_filter_overdetection_count", sleep_filter_overdetection_count)
                print("sleep_filter_sleep_overdetection_count", sleep_filter_sleep_overdetection_count)
                print("sleep_filter_detection_count", sleep_filter_detection_count)

            self.sleep_filtered_offwrist_periods = np.array(self.sleep_filtered_offwrist_periods)
            self.description_filtered_offwrist_periods = np.array(self.description_filtered_offwrist_periods)
        self.sleep_border_overdetection = sleep_border_overdetection


    def refine(self,
               initial_offwrist,
               activity,
               activity_median,
               temperature,
               normalized_temperature_variance,
               temperature_derivative,
               temperature_derivative_variance,
               temperature_threshold,
               ashman_d,
               activity_median_low,
               is_low_temperature_bool,
               filter_half_window_length,
               is_lumus_file,
               additional_information,

               true_offwrist=[],
               true_sleep=[],
               datetime_stamps=[],

               epoch_hour=60,
               long_offwrist_length=4,
               do_near_all_off_detection=True,
               verbose=0,
               dev=False,
               ):
        '''Processes the input data and passes it through the 3-stage
        filtering system to achieve the refined offwrist periods.


        Parameters
        ----------
        initial_offwrist : np.array [int]
                Initial offwrist detection. Based on activity median and
                temperature levels. 1 indicates offwrist, 0 for onwrist
        activity : np.array [float]
                Read activity information
        activity_median : np.array [float]
                Median activity computed from read activity
        temperature : np.array [float]
                Read temperature information
        normalized_temperature_variance : np.array [float]
                Temperature variance computed from read temperature
        temperature_threshold : float
                Estimated temperature threshold that separates offwrist
                and onwrist periods. Temperature levels above this thre-
                shold indicate onwrist
        ashman_d : float
                Statistical measure computed using estimated statistical
                distribution of temperature values
        activity_median_low : np.array [int]
                1 indicates low activity median, 0 otherwise
        filter_half_window_length : int
                Length of the half-window used in the feature extraction 
                filters
        is_lumus_file : boolean
                If True, data was read from ActLumus device
        additional_information : np.array [object]
                Additional information read by device-specific sensors,
                e.g. capacitive sensors in the ActLumus
        true_offwrist : np.array [int]
                Actual offwrist states as marked by the user. 1 indica-
                tes offwrist, 0 for onwrist
        true_sleep : np.array [int]
                Actual sleep states as marked by the user. 1 indicates
                sleep, 0 for awake
        datetime_stamps : np.array [datetime stamp]
                datetime.datetime stamps read by the device
        verbose: int or boolean
                Verbosity level
        dev : boolean
                If True, the algorithm will print and return additional
                information useful for debugging and performance evalua-
                tion 

        Returns
        -------
        refined_offwrist : np.array [int]
                Final refined offwrist detection. 1 indicates offwrist,
                0 for onwrist
        sleep_filtered_offwrist : np.array [int]
                Offwrist detection after sleep filter. 1 indicates off-
                wrist, 0 for onwrist
        is_bimodal : boolean
                If True, data is considered bimodal
        low_temperature_proportion : float
                Proportion of low temperature values in the data
        after_initial_refinement_offwrist_report : pd.DataFrame
                Description report for the offwrist periods before sleep
                filter
        sleep_filtered_report : pd.DataFrame
                Description report for the offwrist periods after sleep
                filter
        filtered_report : pd.DataFrame
                Description report for the offwrist periods after the des-
                cription-based filter
        estimated_sleep : np.array [int]
                Estimated sleep states. 1 for sleep, 0 for awake
        delta_overdetection : int
                Difference between overdetection count before and after
                final filters
        delta_sleep_overdetection : int
                Difference between sleep overdetection count before and 
                after final filters
        delta_detection : int
                Difference between accurate detection count before and 
                after final filters
        '''

        self.initial_offwrist = initial_offwrist
        self.activity = activity
        self.activity_median = activity_median
        self.temperature = temperature
        self.normalized_temperature_variance = normalized_temperature_variance
        self.temperature_derivative = temperature_derivative
        self.temperature_derivative_variance = temperature_derivative_variance
        self.temperature_threshold = temperature_threshold
        self.ashman_d = ashman_d
        self.activity_median_low = activity_median_low
        self.is_low_temperature_bool = is_low_temperature_bool
        self.filter_half_window_length = filter_half_window_length
        self.is_lumus_file = is_lumus_file
        self.additional_information = additional_information
        self.true_offwrist = true_offwrist
        self.true_sleep = true_sleep
        self.datetime_stamps = datetime_stamps
        self.epoch_hour = epoch_hour
        self.long_offwrist_length = long_offwrist_length*epoch_hour
        self.do_near_all_off_detection = do_near_all_off_detection
        # print("verbose",verbose)
        self.verbose = verbose
        # print("self.verbose",self.verbose)
        self.dev = dev

        self.data_length = len(self.initial_offwrist)
        self.refined_offwrist_periods = []
        self.activity_threshold = 0 

        self.datetime_stamps_available = len(self.datetime_stamps) > 0

        self.is_highly_separable = False
        if self.ashman_d > 3:
             self.is_highly_separable = True
        # else:
        #      self.do_valley_peak_filter = True

        self.offwrist_periods = zero_sequences(self.initial_offwrist)
        self.offwrist_count = len(self.offwrist_periods)

        self.delta_overdetection = 0
        self.delta_sleep_overdetection = 0
        self.delta_detection = 0

        if self.offwrist_count > 0:
            self.temperature_variance_threshold = np.quantile(self.normalized_temperature_variance,self.temperature_variance_threshold_quantile,method="linear")                
            self.activity_zero_proportion = zero_prop(self.activity)
            activity_threshold_quantile = self.activity_zero_proportion + (1.0 - self.activity_zero_proportion)*self.activity_threshold_quantile
            self.activity_threshold = np.quantile(self.activity,activity_threshold_quantile,method="linear")

            self.onwrist_periods = zero_sequences(1-self.initial_offwrist)
            self.onwrist_periods_df = self.describe_onwrist_periods()
            self.first_stage_refinement()
            if self.verbose:
                print("first stage onwrist_periods")
                print(self.onwrist_periods)
                print("first stage offwrist_periods")
                self.print_periods(self.offwrist_periods)

            self.second_stage_refinement()

            self.third_stage_refinement()

            self.compute_and_print_results_report()

        else:
            # Trivial case when there are no offwrist periods
            self.refined_offwrist = np.ones(self.data_length)
            self.sleep_filtered_offwrist = refined_offwrist.copy()
            self.is_bimodal = False
            self.low_temperature_proportion = 0.0

        if self.dev:
            return self.refined_offwrist, self.sleep_filtered_offwrist, self.is_bimodal, self.low_temperature_proportion, self.after_initial_refinement_offwrist_report, self.sleep_filtered_report, self.filtered_report, 1-self.estimated_sleep, self.delta_overdetection, self.delta_sleep_overdetection, self.delta_detection

        else:
            return self.refined_offwrist