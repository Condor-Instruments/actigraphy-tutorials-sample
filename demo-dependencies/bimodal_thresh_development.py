"""Temperature bimodal statistical distribution fitting 

Author: Julius A. P. P. de Paula (12/03/2024)
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as scst

from sklearn.mixture import GaussianMixture, BayesianGaussianMixture

# from scipy.optimize import curve_fit

# from unidip import UniDip
# import unidip.dip as dip

import sys, inspect, os

from functions import *


def ashman_d(mu1,sigma1,mu2,sigma2):
    if sigma1+sigma2 > 0:
        return np.sqrt(2.0/(sigma1**2 + sigma2**2))*abs(mu1 - mu2)
    else:
        return 0

def gauss(x,mu,sigma,a):
    return a*np.exp(-(x-mu)**2/2/sigma**2)

def bimodal(x,
            # shape1,
            mu1,sigma1,a1,
            # shape2,
            mu2,sigma2,a2):
    return gauss(x,mu1,sigma1,a1)+gauss(x,mu2,sigma2,a2)

# def unimodal(x,
#             mu1,sigma1,a1,
#             ):
#     return a1*scst.norm.pdf(x,mu1,sigma1)

def bimodal_thresh(data,nbins=100,plot=False,save_plot=False,verbose=False,title="",last_min=False,min_thresh=0.35,dev=False,n_init=5,max_iter=100,tol=1e-2,dthresh=False,fname="bimodalfig.png"):
    n = len(data)
    x = np.linspace(0,1,nbins)

    mu1,sigma1,a1,mu2,sigma2,a2,thresh,dif_thresh = [0,0,0,0,0,0,0,0]
    if n > 1:
        data = np.float64(data)
        counts, bins = np.histogram(data,bins=nbins,range=(0.0,1.0))
        counts_filt = mean_filter(counts,3)

        gm = GaussianMixture(n_components=2, 
                             random_state=0, 
                             n_init=n_init, 
                             max_iter=max_iter, 
                             tol=tol, 
                             verbose=0,
                             init_params="k-means++").fit(data.reshape(-1, 1))

        first = 0
        if gm.means_[0][0] > gm.means_[1][0]:
            first = 1
        mu1,mu2 = [gm.means_[0+first][0],gm.means_[1-first][0]]
        sigma1,sigma2 = [np.sqrt(gm.covariances_[0+first][0][0]),np.sqrt(gm.covariances_[1-first][0][0])]
        a1,a2 = [gm.weights_[0+first]*np.max(counts),gm.weights_[1-first]*np.max(counts)]

        # if verbose:
        #     print(f'len_data={len(data)}')
        #     print(f'mu1={mu1}, mu2={mu2}')
        #     print(f'sigma1={sigma1}, sigma2={sigma2}')
        #     print(f'a1={a1}, a2={a2}')

        params_gm_dict = {"mu1":mu1,"sigma1":sigma1,"a1":a1,"mu2":mu2,"sigma2":sigma2,"a2":a2}
        dist_gm = np.apply_along_axis(bimodal,0,x,**params_gm_dict)
        g1 = np.apply_along_axis(gauss,0,np.linspace(0,1,10000),**{"mu":mu1,"sigma":sigma1,"a":a1})
        g2 = np.apply_along_axis(gauss,0,np.linspace(0,1,10000),**{"mu":mu2,"sigma":sigma2,"a":a2})
        dif = np.absolute(np.subtract(g1,g2))
        start = int(round(10000*(mu1-sigma1)))
        if start < 1500:
            start = 1500
        end = int(round(10000*(mu2+sigma2)))
        if end > 8500:
            end = 8500
        if end < start:
            end = start + 10

        if (mu1-sigma1) < 0.15:
            mu1 = 0.15+sigma1

        if (mu2+sigma2) > 0.85:
            mu2 = 0.85-sigma2

        if mu2 <= mu1:
            mu2 = mu1+2/nbins


        # print(start)
        # print(end)
        
        dif_thresh_id = start+np.argmin(dif[start:end])
        dif_thresh_id = int(round(dif_thresh_id*nbins/10000))
        dif_thresh = bins[dif_thresh_id]

        loc1,loc2 = [np.argmin(np.absolute(bins-mu1)),np.argmin(np.absolute(bins-mu2))]
        if loc2 <= loc1:
            loc2 = loc1+1

        thresh_id_0 = loc1 + np.argmin(dist_gm[loc1:loc2])
        thresh_id_1 = loc1 + np.argmin(counts_filt[loc1:loc2])

        if counts_filt[thresh_id_0] < counts_filt[thresh_id_1]:
            thresh_id = thresh_id_0
        else:
            thresh_id = thresh_id_1

        # if last_min:
        #     while counts_filt[thresh_id+1] == counts_filt[thresh_id]:
        #         thresh_id += 1
        thresh = bins[thresh_id]        

    ash_d = ashman_d(mu1,sigma1,mu2,sigma2)

    # av = counts_filt[thresh_id]
    # if a1 > a2:
    #     bimodal_amplitude = (a2-av)/av
    # else:
    #     bimodal_amplitude = (a1-av)/av

    if verbose:
        print(f'mu1={mu1}, mu2={mu2}')
        print(f'sigma1={sigma1}, sigma2={sigma2}')
        print(f'a1={a1}, a2={a2}')
        print(f'ashman_d={ash_d}')
        # print(f'bimodal_amplitude={bimodal_amplitude}')
        print(f'thresh={thresh}')
        print(f'dif_thresh={dif_thresh}')
        # print((mu1 < nbins/2) and (mu2 < nbins/2))

    # if (ash_d > 1.5) and (ash_d < 2.6):
    #     plot = True

    if (n > 1) and (plot or save_plot):
        plt.figure(figsize=(24,18))
        plt.plot(x,counts,label="counts")
        plt.plot(x,counts_filt,label="counts_filt")
        plt.plot(x,dist_gm,label="dist_gm")
        markerline, stemlines, baseline = plt.stem([thresh_id/nbins,dif_thresh_id/nbins],[counts_filt[thresh_id],counts_filt[dif_thresh_id]],label="divide",linefmt="black",markerfmt="o")
        markerline.set_markerfacecolor('black')
        markerline.set_markersize(10)
        plt.grid()
        plt.legend()
        # plt.title(title+",thresh="+str(thresh)+",dthresh="+str(dif_thresh)+",d="+str(ash_d)+",b="+str(bimodal_amplitude))
        plt.title(title+",thresh="+str(thresh)+",dthresh="+str(dif_thresh)+",d="+str(ash_d))

        if plot and (not save_plot):
            plt.show()
        elif (not plot) and save_plot:
            plt.savefig(fname,bbox_inches="tight")
            plt.close()
        elif plot and save_plot:
            plt.show()
            plt.savefig(fname,bbox_inches="tight")
            plt.close()
        else:
            plt.close()


    if thresh < min_thresh:
        thresh = min_thresh
        if verbose:
            print("clamped to thresh_min")

    if dev:
        if dthresh:
            return dif_thresh, ash_d, thresh
        else:
            return thresh, ash_d, dif_thresh

    else:
        if dthresh:
            return dif_thresh, ash_d
        else:
            return thresh, ash_d