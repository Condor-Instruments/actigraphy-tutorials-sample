import sys
import inspect
import os
import cProfile, pstats, io
import time as ttime

import pandas as pd
import numpy as np


from bimodal_offwrist_refine import BimodalOffwristRefiner
from sklearn.model_selection import ParameterGrid
from scipy.stats import mode


from bimodal_thresh_development import bimodal_thresh
from bimodal_offwrist_feature import bimodal_offwrist_feature
from functions import *

# Features to be extracted during data processing
activity_features = ["signal","median"]
temperature_features = ["signal","var"]
temperature_difference_features = None
capsensor1_features = ['signal']
capsensor2_features = ['signal']

# Numerical parameters
feature_extraction_filters_half_window_length=10 # Half-window length for feature extraction filters
offwrist_activity_quantile=0.15 # Quantile used to compute low activity level
minimum_normalized_activity_threshold=0.015 # Minimum level of the normalized acitivity median to 
                                            # utilize as low-actitvity thresholds
nbins=100 # Related to the precision when computing the low-temperature threshold


trust_grid = {
    "offwrist_maximum_temperature_difference_median":[1.0,0.8,0.85,0.9,0.95,1.05,1.1],
    "offwrist_minimum_temperature_difference_median":[0.75,0.65,0.67,0.69,0.71,0.73,],
    }

lumus_grid = {
    "offwrist_minimum_capsensor1_median":[0.0,0.60,0.65,0.75],
    "offwrist_minimum_capsensor2_median":[0.0,0.60,0.65,0.75,0.8],
    "offwrist_maximum_capsensor1_std":[1.0,0.15,0.12,0.09,],
    "offwrist_maximum_capsensor2_std":[1.0,0.15,0.12,0.09,0.06],
    }

# Refinement sleep estimate filter related parameters
sleep_activity_filter_half_window_length = 120
sleep_low_activity_threshold_configuration = "both"
sleep_all_activity_quantile = 0.4
sleep_positive_activity_quantile = 0.05

do_report_border_concentration_filter = True
do_report_zero_activity_proportion_filter = True
do_report_activity_around_filter = True
do_report_border_activity_filter = True
do_report_low_activity_proportion_filter = True
do_report_low_activity_after = True
report_zero_activity_proportion_minimum = 0.35
border_concentratition_minimum = 0.5
report_activity_around_minimum = 0.1
report_low_activity_proportion_minimum = 0.5

do_description_report_based_filter = True
do_sleep_filter = True
do_statistical_measure_filter = False
do_analyze_sleep_borders = False
temperature_threshold_refinement_intensity = 0.5
temperature_threshold_refinement_quantile = 0.5
bimodal_maximum_offwrist_proportion = 0.0
do_temperature_threshold_refinement = False
do_forbidden_zone = False

do_valley_peak_filter = False
minimum_offwrist_length = 10
bimodal_minimum_low_activity_proportion = 1.0
do_temperature_criteria = False
sleep_low_temperature_proportion_maximum = 0.5

valley_quantile = 0.985
peak_quantile = 0.99
do_valley_peak_algorithm = True

half_day_length_validation = True
do_near_all_off_detection = True 
compute_zero_median_offwrist = True
decrease_ratio_minimum = 0.0

parameter_grid = list(ParameterGrid(lumus_grid))
parameter_grid.append({'offwrist_maximum_capsensor1_std': 1.0, 
                       'offwrist_maximum_capsensor2_std': 1.0, 
                       'offwrist_minimum_capsensor1_median': 0.54, 
                       'offwrist_minimum_capsensor2_median': 0.59})

# neccessary columns: PIM, TEMPERATURE, CAP_SENS_1, CAP_SENS_2, DATE/TIME
def offwrist_wrapper_actlumus(df,verbose=False):
    # config = pd.read_csv(folder+"/offwrist_config.csv",sep=';',header=0,index_col=0)
    # config = config.at[0,"config"]

    refine_verbose = 0
    if verbose:
         refine_verbose += 3.0

    config = len(parameter_grid)-1

    parameter = parameter_grid[config]
    # Parameter initialization
    offwrist_minimum_capsensor1_median = parameter["offwrist_minimum_capsensor1_median"]
    offwrist_minimum_capsensor2_median = parameter["offwrist_minimum_capsensor2_median"]
    offwrist_maximum_capsensor1_std = parameter["offwrist_maximum_capsensor1_std"]
    offwrist_maximum_capsensor2_std = parameter["offwrist_maximum_capsensor2_std"]

    offwrist_maximum_temperature_difference_median = 1.0
    offwrist_minimum_temperature_difference_median = 0.75
    
    data = df[df["TEMPERATURE"] > 0]

    if len(data) > 0:
        datetime_stamps = pd.to_datetime(data['DATE/TIME'])
        datetime_stamps = np.array([to_datetime(date) for date in datetime_stamps])
        epoch_duration = mode(datetime_diff(datetime_stamps),keepdims=True).mode[0]
        epoch_hour = int(3600/epoch_duration)
        half_day_length = 12*epoch_hour
        
        half_window_length = [feature_extraction_filters_half_window_length]
        data = bimodal_offwrist_feature(data,half_window_length,activity_features,temperature_features,temperature_difference_features,capsensor1_features,capsensor2_features,verbose=0)
        data["dt"] = datetime_stamps
        data.sort_values(by="dt",ascending=True,inplace=True)
        if verbose:
            print(data)

        activity = data["activity_signal"].to_numpy()

        data_length = activity.shape[0]

        activity_median = data["activity_median_w="+str(2*feature_extraction_filters_half_window_length+1)].to_numpy()
        normalized_activity_median = norm_01(activity_median)
        activity_median_zero_proportion = zero_prop(normalized_activity_median)
        low_activity_median_quantile = activity_median_zero_proportion + offwrist_activity_quantile*(1-activity_median_zero_proportion)
        low_normalized_activity_median_threshold = np.quantile(normalized_activity_median,low_activity_median_quantile,method='inverted_cdf')
        if low_normalized_activity_median_threshold < minimum_normalized_activity_threshold:
            low_normalized_activity_median_threshold = minimum_normalized_activity_threshold

        is_normalized_activity_median_low_bool = np.where(normalized_activity_median < low_normalized_activity_median_threshold,True,False)
        is_normalized_activity_median_low_int = is_normalized_activity_median_low_bool.astype(int)

        temperature = data["temperature_signal"].to_numpy()
        normalized_temperature = norm_01(temperature)

        temperature_threshold, ashman_d, _ = bimodal_thresh(normalized_temperature[is_normalized_activity_median_low_bool],nbins=nbins,plot=False,save_plot=False,title="temperature",verbose=False,dev=True)
        
        mean_temperature = np.mean(temperature)
        median_high_activity_temperature = np.median(temperature[np.logical_not(is_normalized_activity_median_low_bool)])
        
        # Rescaling normalized temperature threshold
        temperature_minimum = np.min(temperature) 
        temperature_maximum = np.max(temperature)
        temperature_threshold = temperature_minimum + temperature_threshold*(temperature_maximum - temperature_minimum)
        if temperature_threshold > median_high_activity_temperature:
                temperature_threshold = median_high_activity_temperature

        is_low_temperature_bool = np.where(temperature < temperature_threshold,True,False)
        
        normalized_temperature_variance = norm_01(data["temperature_var_w="+str(2*feature_extraction_filters_half_window_length+1)].to_numpy())

        temperature_derivative = diff5(data["temperature_signal"])
        temperature_derivative_variance = var_filter(temperature_derivative,feature_extraction_filters_half_window_length)

        offwrist_estimate = np.where(np.logical_and(is_low_temperature_bool,is_normalized_activity_median_low_bool),0,1)

        rolling_quantile = 0.7
        activity_rolling_quantile = quantile_filter(activity,feature_extraction_filters_half_window_length,quantile=rolling_quantile,method='inverted_cdf')
        zero_activity_median = np.where(activity_rolling_quantile == 0,0,1)
        
        zero_median_offwrists = zero_sequences(zero_activity_median,minimum_length=half_day_length)
        zero_median_offwrist = np.ones(data_length)
        if compute_zero_median_offwrist:
            for off in zero_median_offwrists:
                zero_median_offwrist[off[0]:off[1]] = 0.0

        lumus = True
        do_temperature_difference_filter = False
        do_capacitive_sensor_variance_filter = True
        capsensor1 = norm_01(data["capsensor1_signal"].to_numpy())
        capsensor2 = norm_01(data["capsensor2_signal"].to_numpy())
        capsensor_dif = np.subtract(capsensor1,capsensor2)
        additional_information = {"c1":capsensor1,"c2":capsensor2,"capsensor_dif":capsensor_dif}
        
        refiner = BimodalOffwristRefiner(minimum_offwrist_length=minimum_offwrist_length,
                                             do_description_report_based_filter=do_description_report_based_filter,do_sleep_filter=do_sleep_filter,do_temperature_difference_filter=do_temperature_difference_filter,
                                             sleep_activity_filter_half_window_length=sleep_activity_filter_half_window_length,sleep_low_activity_threshold_configuration=sleep_low_activity_threshold_configuration,
                                             sleep_all_activity_quantile=sleep_all_activity_quantile,sleep_positive_activity_quantile=sleep_positive_activity_quantile,do_capacitive_sensor_variance_filter=do_capacitive_sensor_variance_filter,
                                             do_report_activity_around_filter=do_report_activity_around_filter,do_report_border_concentration_filter=do_report_border_concentration_filter,
                                             do_report_border_activity_filter=do_report_border_activity_filter,do_report_zero_activity_proportion_filter=do_report_zero_activity_proportion_filter,
                                             report_zero_activity_proportion_minimum=report_zero_activity_proportion_minimum,border_concentratition_minimum=border_concentratition_minimum,
                                             report_activity_around_minimum=report_activity_around_minimum,do_statistical_measure_filter=do_statistical_measure_filter,do_analyze_sleep_borders=do_analyze_sleep_borders,
                                             do_valley_peak_algorithm=do_valley_peak_algorithm,peak_quantile=peak_quantile,valley_quantile=valley_quantile,do_temperature_threshold_refinement=do_temperature_threshold_refinement,
                                             temperature_threshold_refinement_intensity=temperature_threshold_refinement_intensity,temperature_threshold_refinement_quantile=temperature_threshold_refinement_quantile,
                                             do_report_low_activity_proportion_filter=do_report_low_activity_proportion_filter,report_low_activity_proportion_minimum=report_low_activity_proportion_minimum,
                                             do_report_low_activity_after=do_report_low_activity_after,
                                             do_forbidden_zone=do_forbidden_zone,
                                             do_valley_peak_filter=do_valley_peak_filter,
                                             bimodal_maximum_offwrist_proportion=bimodal_maximum_offwrist_proportion,
                                             bimodal_minimum_low_activity_proportion=bimodal_minimum_low_activity_proportion,
                                             do_temperature_criteria=do_temperature_criteria,
                                             sleep_low_temperature_proportion_maximum=sleep_low_temperature_proportion_maximum,
                                             offwrist_maximum_temperature_difference_median=offwrist_maximum_temperature_difference_median,
                                             offwrist_minimum_temperature_difference_median=offwrist_minimum_temperature_difference_median,
                                             offwrist_minimum_capsensor1_median=offwrist_minimum_capsensor1_median,
                                             offwrist_minimum_capsensor2_median=offwrist_minimum_capsensor2_median,
                                             offwrist_maximum_capsensor1_std=offwrist_maximum_capsensor1_std,
                                             offwrist_maximum_capsensor2_std=offwrist_maximum_capsensor2_std,
                                             half_day_length_validation=half_day_length_validation,
                                             decrease_ratio_minimum=decrease_ratio_minimum,
                                             )
        refined_offwrist_estimate = refiner.refine(offwrist_estimate,
                                                    activity,activity_median,
                                                    temperature,normalized_temperature_variance,temperature_derivative,temperature_derivative_variance,
                                                    temperature_threshold,ashman_d,
                                                    is_normalized_activity_median_low_int,is_low_temperature_bool,
                                                    feature_extraction_filters_half_window_length,
                                                    lumus,additional_information,
                                                    epoch_hour=epoch_hour,
                                                    verbose=refine_verbose,
                                                    datetime_stamps=datetime_stamps,
                                                    do_near_all_off_detection=do_near_all_off_detection,)        
        final_offwrist = zero_median_offwrist*refined_offwrist_estimate

        out = np.zeros(len(df)) 
        out[np.where(df["TEMPERATURE"] > 0,True,False)] = final_offwrist

    else:
        out = np.ones(len(df))
    
    return 4*(1.0-out)
