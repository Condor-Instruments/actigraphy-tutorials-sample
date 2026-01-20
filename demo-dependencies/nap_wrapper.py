import sys
import inspect
import os

import numpy as np

from cspd import CSPD as cr


# param_set = [5.00000000e-01, 1.36818215e-01, 2.83303006e-01, 9.76738814e-01,
# 5.00000000e-01, 1.00000000e-06, 5.00000000e-01, 1.00000000e-06,
# 1.00000000e-06, 1.00000000e-06, 5.00000000e-01, 1.00000000e-06,

# 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+01,

# 7.01515672e-01, 5.00000000e-02, 1.03437162e-01, 

# 6.75000000e-01, 5.00000000e-01, 2.0, 1.50000000e+01, 0.6, 0.5, 7.0,

# 1.00000000e+00, 0.00000000e+00, 0.00000000e+00,
# 1.00000000e+00, 4.75000000e-01, 1.00000000e+00, 0.00000000e+00,
# 1.00000000e+00, 6.00000000e-01]

param_set = [0.21052689,  0.84210458,  0.10526395,  0.21052689,  0.21052689,  0.57894721,
 0.63157868,  0.63157868,  0.57894721,  0.68421016,  0.89473605,  0.68421016,

12.        , 16.        ,  7.        , 12.        ,  
  
 0.67210526,  0.43842105,  0.58421053,  

 0.97894737,  0.05263247,  3.15789563, 17.        ,  0.45,        0.22105263,  4.,

 1.        ,  1.        ,  0.        ,  0.,          0.50263158,  
 1.        ,  0.        ,  1.        ,  0.41578947,]

def nap_wrapper(df,verbose=False):
    nap_bool = np.where(df["state"].to_numpy() == 0, True, False)   # 0 means awake
    #                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          median_filter_window_hourly_length,                   adaptive_median_filter_padding_hourly_length,                 nt,              lp,                qt,                  nzt                                                                                                                             
    # b1, b2, b3, b4, b5, b6, g1, g2, g3, g4, g5, g6, l1, l2, l3, l4, c1, c2, c3, median_filter_window_hourly_length, adaptive_median_filter_padding_hourly_length, nt, lp, qt,nzt = [0.3926282123537868, 0.973726216816406, 0.6412124219518207, 0.06979307508732235, 0.5450946664202019, 0.9884188094422669, 0.37132795028895893, 0.1432054193228914, 0.701760011480025, 0.38437495589948734, 0.9183860823042017, 0.6479220057947142, 4.543733212486201, 18.890174446464872, 13.957756725136454, 4.793390507901703, 0.4795715360406394, 0.5805597857294332, 0.012428473364671828, 3.8381319041667505, 2.633529715744524, 6.592131249520484, 18.00761071873817, 0.02681760463120241, 0.9997753895348146]
    # b1, b2, b3, b4, b5, b6, g1, g2, g3, g4, g5, g6, l1, l2, l3, l4, c1, c2, c3, median_filter_window_hourly_length, adaptive_median_filter_padding_hourly_length, nt, lp, qt,nzt = [0.515387705140748, 0.1368182152155934, 0.2833030057430186, 0.9767388135080795, 0.6414219543156179, 0.020819747009962, 0.7481753894274739, 0.9964924902547526, 0.05265783544827686, 0.1464248884092994, 0.6641912735396098, 0.7919161220074059, 4.930033092546711, 3.9120572096034474, 15.94050372428606, 10.962941213073494, 0.7015156722371547, 0.2290212023098121, 0.10343716196493354, 1.9295818046503144, 0.9066224948419366, 5.733928826022829, 22.12513640657167, 0.10935805258701688, 0.7124802345986866]
    # b1, b2, b3, b4, b5, b6, g1, g2, g3, g4, g5, g6, l1, l2, l3, l4, c1, c2, c3, median_filter_window_hourly_length, adaptive_median_filter_padding_hourly_length, nt, lp, qt,nzt = [0.09776991327058086, 0.6545417479079847, 0.5451187619723755, 0.5230331131532031, 0.0783897291118946, 0.789560241940211, 0.8958440162451147, 0.35996589522824074, 0.7409851985177116, 0.6544547656442564, 0.04462276043383443, 0.8179057199662296, 2.6389992899596715, 14.410549445358408, 19.710145761848207, 6.367079267329911, 0.2991509491616431, 0.42767234041620894, 0.19577234390888892, 1.1338195428822526, 0.253813835054061, 8.884327944169857, 11.797091995546275, 0.6531810610626234, 0.6946561509731992]


    b1,b2,b3,b4,b5,b6,g1,g2,g3,g4,g5,g6,l1,l2,l3,l4,c1,c2,c3,median_filter_window_hourly_length,adaptive_median_filter_padding_hourly_length,nt,lp,qt,nzt,peak_valley_minimum_length,bedtime_do_remove_before_long_peak,bedtime_do_remove_before_tall_peak,bedtime_do_remove_after_long_valley,bedtime_score_last_candidate,bedtime_metric_parameter,getuptime_do_remove_after_long_tall_peak,getuptime_do_remove_before_long_valley,getuptime_score_first_candidate,getuptime_metric_parameter = param_set

    b = [b1, b2, b3, b4, b5, b6,]
    g = [g1, g2, g3, g4, g5, g6,]


    l = [l1, l2, l3, l4,]
    l = [int(round(le)) for le in l]
    lp = int(round(lp))
    c = [c1, c2, c3]

    cresp = cr(median_filter_window_hourly_length=median_filter_window_hourly_length,
               adaptive_median_filter_padding_hourly_length=adaptive_median_filter_padding_hourly_length,
               preprocessing_morphological_filter_structuring_element_size=lp, # Useless
               detect_naps=True,
               nap_median_activity_threshold=nt,
               nap_zero_proportion_threshold=nzt,
               compute_output_naps_with_logical_and=True,
               
               do_peak_valley_length_filter=True,
               peak_valley_minimum_length=peak_valley_minimum_length,
               median_filter_short_window=20,
               after_candidate_window=15,
               activity_median_analysis_window=20,

               bedtime_scores=b,
               getuptime_scores=g,
               length_thresholds=l,
               candidate_thresholds=c,
               short_window_activity_median_threshold_quantile=qt,
                
               getuptime_high_probability_awake_peak_length=20,
               getuptime_high_probability_sleep_valley_length=20,

               bedtime_high_probability_awake_peak_length=20,
               bedtime_high_probability_sleep_valley_length=20,

               bedtime_do_remove_before_long_peak=bedtime_do_remove_before_long_peak,
               bedtime_do_remove_before_tall_peak=bedtime_do_remove_before_tall_peak,
               bedtime_do_remove_after_long_valley=bedtime_do_remove_after_long_valley,
               bedtime_score_last_candidate=bedtime_score_last_candidate,
               bedtime_metric_parameter=bedtime_metric_parameter,
               bedtime_update_peaks_and_valleys=False,
               bedtime_do_bedtime_candidates_crossings_filter=False,
               bedtime_consider_second_best_candidate=False,

               getuptime_do_remove_after_long_tall_peak=getuptime_do_remove_after_long_tall_peak,
               getuptime_do_remove_before_long_valley=getuptime_do_remove_before_long_valley,
               getuptime_score_first_candidate=getuptime_score_first_candidate,
               getuptime_metric_parameter=getuptime_metric_parameter,
               getuptime_update_peaks_and_valleys=False,
               )
    cresp.model(
                df["activity"].to_numpy()[nap_bool],
                df["datetime"].to_numpy()[nap_bool],
                verbose=verbose,
                )

    state = np.where(nap_bool,0,df["state"].to_numpy())
    state[nap_bool] = 1-cresp.refined_output   # 0 means awake and 1 means sleeping

    activity_zero_proportions = np.zeros(len(state))
    activity_zero_proportions[nap_bool] = cresp.activity_zero_proportions

    thresholded_activity_zero_proportions = np.zeros(len(state))
    thresholded_activity_zero_proportions[nap_bool] = cresp.thresholded_activity_zero_proportions

    thresholded_median_activity = np.zeros(len(state))
    thresholded_median_activity[nap_bool] = cresp.thresholded_median_activity

    return state
