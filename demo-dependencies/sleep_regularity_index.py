import sys
import os
import inspect

import pandas as pd
import numpy as np
from datetime import date,datetime,timedelta,time

# root = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# sys.path.insert(0, root + '/pylogread')
from check_consistency import consistency_check

def actigraphy_calculate_SRI(df,sleep,epoch=60,visualize=False,verbose=False):
    consistency = consistency_check(df,epoch)
    desc = consistency["desc"].to_numpy()

    regular_epochs = 0
    num_epochs = 0
    sri = 0

    if not (("backward" in desc) or ("1970" in desc) or ("2000" in desc)):    
        last = df.index.to_series().iat[-1]

        if visualize:
            visual = pd.DataFrame([],columns=["di","si","di_","si_","delta"])

        for m in df.index:
            si = df.at[m,sleep]
            next_day = m+pd.Timedelta(days=1)
            if next_day <= last: 
                if next_day in df.index:  # pode haver pequeno gap
                    si_ = df.at[next_day,sleep]
                    delta = 0
                    if si_ == si:
                        regular_epochs += 1
                        delta = 1

                    num_epochs += 1

                    if visualize:
                        v = pd.DataFrame([[m,si,next_day,si_,delta]],columns=["di","si","di_","si_","delta"])
                        visual = pd.concat([visual,v],ignore_index=True)
            else:
                break

        if num_epochs > 0:
            sri = -100 + 200*(regular_epochs/num_epochs)

        if verbose:
            print("num_epochs",num_epochs)
            print("regular_epochs",regular_epochs)
            print("sri",sri)

        if visualize:
            visual.to_csv("visualize.csv",header=True,index=False)
    else:
        print("inconsistent data, could not compute SRI")

    return sri