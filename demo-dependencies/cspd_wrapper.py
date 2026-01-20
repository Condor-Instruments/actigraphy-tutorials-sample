import sys
import inspect
import os
import numpy as np


from cspd import CSPD as cd

param_set = [0.4081634489795918,0.551020306122449,0.7142852857142857,0.6122446734693877,0.48979593877551014,0.44897969387755093,0.26530659183673466,0.48979593877551014,0.795917775510204,0.24489846938775509,0.999999,0.24489846938775509,8.0,20.0,17.0,16.0,0.4328571428571429,0.37244897959183676,0.5619047619047619,0.43163269387755104,11.0,1.0,0.0,0.0,1.0,0.42755102040816323,41.0,47.0,28.0,7.0,9.0,9.0,1.0,0.0,1.0,0.4071428571428571]

b1,b2,b3,b4,b5,b6,g1,g2,g3,g4,g5,g6,l1,l2,l3,l4,c1,c2,c3,qt,peak_valley_minimum_length,bedtime_do_remove_before_long_peak,bedtime_do_remove_before_tall_peak,bedtime_do_remove_after_long_valley,bedtime_score_last_candidate,bedtime_metric_parameter,median_filter_short_window,after_candidate_window,half_window_around_border,median_filter_half_window_size,short_window_activity_median_minimum_high_epochs,activity_median_analysis_window,getuptime_do_remove_after_long_tall_peak,getuptime_do_remove_before_long_valley,getuptime_score_first_candidate,getuptime_metric_parameter = param_set
b = [b1, b2, b3, b4, b5, b6,]
g = [g1, g2, g3, g4, g5, g6,]
l = [l1, l2, l3, l4,]
l = [int(round(ln)) for ln in l]
c = [c1, c2, c3]

def cspd_wrapper(df,verbose=False,dev=False):
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

    Returns
    -------
    activity : np.array [float]
            input activity array
    """
    onwrist = np.where(df["state"].to_numpy() == 4, False, True)

    cspd = cd(do_peak_valley_length_filter=True,
              sleep_median_activity_quantile_threshold=0.365,

              peak_valley_minimum_length=peak_valley_minimum_length,
              median_filter_short_window=median_filter_short_window,
              after_candidate_window=after_candidate_window,
              half_window_around_border=half_window_around_border,
              median_filter_half_window_size=median_filter_half_window_size,
              short_window_activity_median_minimum_high_epochs=short_window_activity_median_minimum_high_epochs,
              activity_median_analysis_window=activity_median_analysis_window,
           
              bedtime_scores=b,
              getuptime_scores=g,
              length_thresholds=l,
              candidate_thresholds=c,
              short_window_activity_median_threshold_quantile=qt,

              getuptime_high_probability_awake_peak_length=45,
              getuptime_high_probability_sleep_valley_length=45,

              bedtime_high_probability_awake_peak_length=45,
              bedtime_high_probability_sleep_valley_length=45,

              bedtime_do_remove_before_long_peak=bedtime_do_remove_before_long_peak,
              bedtime_do_remove_before_tall_peak=bedtime_do_remove_before_tall_peak,
              bedtime_do_remove_after_long_valley=bedtime_do_remove_after_long_valley,
              bedtime_update_peaks_and_valleys=False,
              bedtime_do_bedtime_candidates_crossings_filter=False,
              bedtime_consider_second_best_candidate=True,
              bedtime_score_last_candidate=bedtime_score_last_candidate,
              bedtime_metric_parameter=bedtime_metric_parameter,
              
              getuptime_do_remove_after_long_tall_peak=getuptime_do_remove_after_long_tall_peak,
              getuptime_do_remove_before_long_valley=getuptime_do_remove_before_long_valley,
              getuptime_score_first_candidate=getuptime_score_first_candidate,
              getuptime_update_peaks_and_valleys=False,
              getuptime_metric_parameter=getuptime_metric_parameter,
              )
    
    cspd.model(df["activity"].to_numpy()[onwrist],
               df["datetime"].to_numpy()[onwrist],
               verbose=verbose)

    state = np.where(onwrist,0,4)
    state[onwrist] = 1-cspd.refined_output

    edge_median = np.zeros(len(state))
    edge_median[onwrist] = cspd.refinement_window_median

    edge_metric = np.zeros(len(state))
    edge_metric[onwrist] = cspd.refinement_window_metric_threshold

    edge_median_diff = np.zeros(len(state))
    edge_median_diff[onwrist] = cspd.refinement_window_median_difference

    xf_aux = np.zeros(len(state))
    xf_aux[onwrist] = cspd.short_window_activity_median

    edge_levels = np.zeros(len(state))
    edge_levels[onwrist] = cspd.refinement_window_levels

    cspd_y2 = np.zeros(len(state))
    cspd_y2[onwrist] = cspd.improved_sleep_detection

    cspd_y2 = np.zeros(len(state))
    cspd_y2[onwrist] = cspd.improved_sleep_detection

    cspd_xfa = np.zeros(len(state))
    cspd_xfa[onwrist] = cspd.adaptive_median_filtered_activity

    cspd_sq = cspd.sleep_median_activity_quantile_threshold*np.ones(len(state))

    if dev:        
        # return state,cspd.padded_mitigated_zeros_activity,cspd.initial_sleep_detection,cspd.morphological_filtered_initial_detection,cspd_xfa,cspd_y2
        return state, cspd_y2, cspd_xfa, cspd_sq #,edge_median,edge_metric,edge_levels,cspd.short_window_activity_median_threshold,xf_aux #,edge_median_diff
    
    else:
        return state