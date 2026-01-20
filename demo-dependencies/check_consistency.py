import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

def datetime_diff(stamps):
    datetime_diff = np.array([(stamps[i]-stamps[i-1]).total_seconds() for i in range(1,len(stamps))])
    datetime_diff = np.insert(datetime_diff,0,[0])
    return datetime_diff

def check_1970(time_index):
    n = len(time_index)
    check_1970 = []
    if n > 1:
        year = time_index.year
        check_1970 = [["1970",i,time_index[i]] for i in range(len(year)) if year[i] == 1970]
    check_1970 = pd.DataFrame(check_1970,columns=["desc","index","stamp"])
    # print(check_1970)
    return check_1970

def check_2000(time_index):
    n = len(time_index)
    check_2000 = []
    if n > 1:
        year = time_index.year
        check_2000 = [["2000",i,time_index[i]] for i in range(len(year)) if year[i] == 2000]
    check_2000 = pd.DataFrame(check_2000,columns=["desc","index","stamp"])
    # print(check_2000)
    return check_2000

def check_backward(time_index):
    n = len(time_index)
    check_backward = []
    if n > 1:
        diff = datetime_diff(time_index)
        check_backward = [["backward",diff[i],i,time_index[i-1],time_index[i]] for i in range(len(diff)) if diff[i] < 0]
    check_backward = pd.DataFrame(check_backward,columns=["desc","gap","index","from","to"])

    # print(check_backward)
    return check_backward

def check_gap(time_index,duration):
    n = len(time_index)
    check_gap = []
    if n > 1:
        diff = datetime_diff(time_index)
        check_gap = [["gap",diff[i],i,time_index[i-1],time_index[i]] for i in range(len(diff)) if diff[i] > duration]
    check_gap = pd.DataFrame(check_gap,columns=["desc","gap","index","from","to"])
    # print(check_gap)
    return check_gap

def consistency_check(data,duration):
    consistency = pd.concat([check_1970(data.index),check_2000(data.index),check_backward(data.index),check_gap(data.index,duration)],ignore_index=True)
    consistency["misc"] = np.nan
    consistency["ignore"] = False
    consistency.sort_values("index",ignore_index=True,inplace=True)

    return consistency