"""Feature extraction for bimodal offwrist algorithm 

Author: Julius A. P. P. de Paula (--/--/2023)
"""

import sys
import inspect
import os
import cProfile, pstats, io
import time as ttime

import pandas as pd
import numpy as np


from functions import *

def bimodal_offwrist_feature(df,half_window_length,activity_features,temperature_features,temperature_difference_features,capsensor1_features,capsensor2_features,verbose=0):
        """Extracts features from PIM, internal temperature and capacitive sensors
           data read by the ActTrust device

        Parameters
        ----------
        df : pd.DataFrame
                Information read by the ActTrust
        half_window_length : np.array [int]
                Window lengths to use in the feature extraction filters
        activity_features : np.array [string]
                Features to be extracted from PIM information
        temperature_features : np.array [string]
                Features to be extracted from internal temperature information
        capsensor1_features : np.array [string]
                Features to be extracted from capacitive sensor 1 information
        capsensor2_features : np.array [string]
                Features to be extracted from capacitive sensor 2 information
        verbose: {0,1,2}
                Verbosity level

        Returns
        -------
        data : pd.DataFrame
                Extracted features from all readings
        """

        start_time_ = ttime.time()


        # Extracting features from PIM
        if activity_features == None:
                data = pd.DataFrame([])
        else:
                pim = df["PIM"].to_numpy()
                start_time = ttime.time()    
                data = extract_features(pim,half_window_length=half_window_length,column_prefix="activity_",features=activity_features)
                if verbose:
                        print("lg pim ",ttime.time()-start_time,"\n")
        if verbose > 1:
                print(data)

        # Extracting features from internal temperature
        if temperature_features != None:
                int_temp = df["TEMPERATURE"].to_numpy()
                start_time = ttime.time()
                int_temp_data = extract_features(int_temp,half_window_length=half_window_length,column_prefix='temperature_',features=temperature_features)
                data = pd.concat([data,int_temp_data],axis=1,ignore_index=False)
                if verbose:
                        print("lg temp ",ttime.time()-start_time,"\n")
                if verbose > 1:
                        print(lg.data)
        if verbose > 1:
                print(data)

        
        if temperature_difference_features != None:
                ext_temp = df["EXT TEMPERATURE"].to_numpy()
                if np.sum(ext_temp) == 0:
                        ext_temp = df["TEMPERATURE"].to_numpy()-0.6

                dif_temp = np.absolute(np.subtract(int_temp,ext_temp))
                start_time = ttime.time()
                dif_data = extract_features(dif_temp,half_window_length=half_window_length,column_prefix='diftemp_',features=temperature_difference_features)
                dif_data["ext_temp"] = ext_temp
                data = pd.concat([data,dif_data],axis=1,ignore_index=False)
                if verbose:
                        print("lg dif ",ttime.time()-start_time,"\n")
                if verbose > 1:
                        print(lg.data)
        if verbose > 1:
                print(data)


        # Extracting features from capacitive sensor 1
        if capsensor1_features != None:
                if "CAP_SENS_1" in df.columns:
                        cap1 = norm_01(df["CAP_SENS_1"].to_numpy())
                        cap1 = df["CAP_SENS_1"].to_numpy()

                        start_time = ttime.time()
                        cap1_data = extract_features(cap1,half_window_length=half_window_length,column_prefix='capsensor1_',features=capsensor1_features)
                        data = pd.concat([data,cap1_data],axis=1,ignore_index=False)
                        if verbose:
                                print("lg c1 ",ttime.time()-start_time,"\n")
                        if verbose > 1:
                                print(lg.data)
                else:
                        if verbose:
                                print("no cap1 column\n")
        if verbose > 1:
                print(data)

        # Extracting features from capacitive sensor 2
        if capsensor2_features != None:
                if "CAP_SENS_2" in df.columns:
                        cap2 = norm_01(df["CAP_SENS_2"].to_numpy())
                        cap2 = df["CAP_SENS_2"].to_numpy()

                        start_time = ttime.time()                        
                        cap2_data = extract_features(cap2,half_window_length=half_window_length,column_prefix='capsensor2_',features=capsensor2_features)
                        data = pd.concat([data,cap2_data],axis=1,ignore_index=False)
                        if verbose:
                                print("lg c2 ",ttime.time()-start_time,"\n")
                        if verbose > 1:
                                print(lg.data)
                else:
                        if verbose:
                                print("no cap2 column\n")
        if verbose > 1:
                print(data)

        if verbose:
                print("total ",ttime.time()-start_time_,"\n")

        return data