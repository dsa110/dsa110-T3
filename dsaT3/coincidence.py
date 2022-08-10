import os
import sys

import time
import numpy as np
import matplotlib.pylab as plt
import glob
import pandas as pd 

from astropy import units as u 
from astropy.time import Time
from sigpyproc.Readers import FilReader

import staretools 

def rsync_heimdall_cand():
    os.system("rsync -avzhe ssh user@158.154.14.10:/home/user/cand_times_sync/ /home/user/cand_times_sync");
    os.system("rsync -avzhe ssh user@166.140.120.248:/home/user/cand_times_sync/ /home/user/cand_times_sync_od");
    time.sleep(60)

def coincidence_2pt(mjd_arr_1, mjd_arr_2, t_thresh_sec=0.25):
    n1 = len(mjd_arr_1)
    n2 = len(mjd_arr_2)
    coincidence_arr = []

    if n1<=n2:
        for ii in range(n1):
#            if ii % 1000==0:
#                print("%d/%d"%(ii,n1))
            min_ind = np.argmin(np.abs(mjd_arr_1[ii]-mjd_arr_2))
            min_t = np.abs(mjd_arr_1[ii] - mjd_arr_2[min_ind])*86400
            if min_t < t_thresh_sec:
                coincidence_arr.append((ii, min_ind, min_t))

        return np.array(coincidence_arr)

    if n2<n1:
        for ii in range(n2):
#            if ii % 1000==0:
#                print("%d/%d"%(ii,n2))
            min_ind = np.argmin(np.abs(mjd_arr_2[ii]-mjd_arr_1))
            min_t = np.abs(mjd_arr_2[ii] - mjd_arr_1[min_ind])*86400
            if min_t < t_thresh_sec:
                coincidence_arr.append((min_ind, ii, min_t))

        return np.array(coincidence_arr)

def get_coincidence_3stations(fncand1, fncand2, fncand3, 
                              t_thresh_sec=0.25, 
                              nday_lookback=1., 
                              dm_diff_thresh=5.0,
                              dm_frac_thresh=0.07):
    """ Read in .cand files, convert to pandas DataFrame,
    look for coincidences in time between all three stations
    as well as coincidences in time AND dm between all pairs 
    of stations. Return tuple with coincident triggers.
    """

    # Read in candidates
    data_1 = staretools.read_heim_pandas(fncand1, skiprows=0)
    data_2 = staretools.read_heim_pandas(fncand2, skiprows=0)
    data_3 = staretools.read_heim_pandas(fncand3, skiprows=0)

    mjd_arr_1 = staretools.get_mjd_cand_pd(data_1).values
    mjd_arr_2 = staretools.get_mjd_cand_pd(data_2).values

    try:
        mjd_arr_3 = staretools.get_mjd_cand_pd(data_3).values
    except:
        print("Could not read fncand3 MJDs as pandas df")
        mjd_arr_3 = staretools.get_mjd_cand(fncand3)

    # Find candidates that happened in past nday_lookback days
    ind1 = np.where(mjd_arr_1>(Time.now().mjd-nday_lookback))[0]
    ind2 = np.where(mjd_arr_2>(Time.now().mjd-nday_lookback))[0]
    ind3 = np.where(mjd_arr_3>(Time.now().mjd-nday_lookback))[0]

    data_1, mjd_arr_1 = data_1.loc[ind1], mjd_arr_1[ind1]
    data_2, mjd_arr_2 = data_2.loc[ind2], mjd_arr_2[ind2]
    data_3, mjd_arr_3 = data_3.loc[ind3], mjd_arr_3[ind3]

    first_MJD = min(mjd_arr_1)
    last_MJD = max(mjd_arr_1)
    print("\nSearching for coincidences between %0.5f and %0.5f\n" % (first_MJD, last_MJD))

    if len(ind1)>0 and len(ind2)>0:
        print("Starting %s"%fncand1)
        coincidence_arr12 = coincidence_2pt(mjd_arr_1, mjd_arr_2, 
                                        t_thresh_sec=t_thresh_sec)
        print("Found %d coincidences station: 1x2"%len(coincidence_arr12))
    else:
        coincidence_arr12 = []
        print("No candidates in past %d days 1x2" % nday_lookback)

    if len(ind2)>0 and len(ind3)>0: 
        print("Starting %s"%fncand2)
        coincidence_arr23 = coincidence_2pt(mjd_arr_2, mjd_arr_3, 
                                        t_thresh_sec=t_thresh_sec)
        print("Found %d coincidences station: 2x3"%len(coincidence_arr23))
    else:
        coincidence_arr23 = []
        print("No candidates in past %d days 2x3" % nday_lookback)

    if len(ind1)>0 and len(ind3)>0:
        print("Starting %s"%fncand3)
        coincidence_arr13 = coincidence_2pt(mjd_arr_1, mjd_arr_3, 
                                        t_thresh_sec=t_thresh_sec)
        print("Found %d coincidences station: 1x3"%len(coincidence_arr13))
    else:
        coincidence_arr13 = []
        print("No candidates in past %d days 1x3" % nday_lookback)

    if len(coincidence_arr12):
        print("Finding time/DM coincidences stations 1x2")
        dm_1_12 = data_1.iloc[coincidence_arr12[:,0].astype(int)]['dm']
        dm_2_12 = data_2.iloc[coincidence_arr12[:,1].astype(int)]['dm']       
        abs_diff = np.abs(dm_1_12.values-dm_2_12.values)
        frac_diff = abs_diff/(0.5*(dm_1_12.values+dm_2_12.values)) 
        ind_ = np.where((abs_diff<dm_diff_thresh) | (frac_diff<dm_frac_thresh))[0]
        ind_12 = (coincidence_arr12[:,0].astype(int))[ind_]
        ind_21 = (coincidence_arr12[:,1].astype(int))[ind_]

        # Now take the mean time for each pair.
        mjd_arr_12 = 0.5*(mjd_arr_1[coincidence_arr12[:,0].astype(int)]+\
                     mjd_arr_2[coincidence_arr12[:,1].astype(int)])

        print("Finding time coincidences stations 1x2x3")
        coincidence_arr123 = coincidence_2pt(mjd_arr_12, mjd_arr_3, 
                                            t_thresh_sec=t_thresh_sec)

    else:
        ind_12, ind_21 = [],[]
        coincidence_arr123 = []
        mjd_arr_12 = []

    if len(coincidence_arr23):
        print("Finding time/DM coincidences stations 2x3")
        dm_2_23 = data_2.iloc[coincidence_arr23[:,0].astype(int)]['dm']
        dm_3_23 = data_3.iloc[coincidence_arr23[:,1].astype(int)]['dm']
        abs_diff = np.abs(dm_2_23.values-dm_3_23.values)
        frac_diff = abs_diff/(0.5*(dm_2_23.values+dm_3_23.values)) 
        ind_ = np.where((abs_diff<dm_diff_thresh) | (frac_diff<dm_frac_thresh))[0]
        ind_23 = (coincidence_arr23[:,0].astype(int))[ind_]
        ind_32 = (coincidence_arr23[:,1].astype(int))[ind_]


        mjd_arr_23 = 0.5*(mjd_arr_2[coincidence_arr23[:,0].astype(int)]+\
                          mjd_arr_3[coincidence_arr23[:,1].astype(int)])

        coincidence_arr231 = coincidence_2pt(mjd_arr_23, mjd_arr_1, 
                                            t_thresh_sec=t_thresh_sec)
    else:
        ind_23, ind_32 = [],[]
        coincidence_arr231 = []
        mjd_arr_23 = []

    if len(coincidence_arr13):
        print("Finding time/DM coincidences stations 1x3")
        dm_1_13 = data_1.iloc[coincidence_arr13[:,0].astype(int)]['dm']
        dm_3_13 = data_3.iloc[coincidence_arr13[:,1].astype(int)]['dm']
        abs_diff = np.abs(dm_1_13.values-dm_3_13.values)
        frac_diff = abs_diff/(0.5*(dm_1_13.values+dm_3_13.values))
        ind_ = np.where((abs_diff<dm_diff_thresh) | (frac_diff<dm_frac_thresh))[0]
        ind_13 = (coincidence_arr13[:,0].astype(int))[ind_]
        ind_31 = (coincidence_arr13[:,1].astype(int))[ind_]

        mjd_arr_13 = 0.5*(mjd_arr_1[coincidence_arr13[:,0].astype(int)]+\
                          mjd_arr_3[coincidence_arr13[:,1].astype(int)])

        print("Finding time coincidences stations 3x1x2")
        coincidence_arr312 = coincidence_2pt(mjd_arr_3, mjd_arr_12, 
                                            t_thresh_sec=t_thresh_sec)
    else:
        ind_13, ind_31 = [],[]
        coincidence_arr312 = []
        mjd_arr_13 = []

    if len(coincidence_arr123):
        ind_1_3x = coincidence_arr12[:,0].astype(int)[coincidence_arr123[:,0].astype(int)]
        ind_2_3x = coincidence_arr12[:,1].astype(int)[coincidence_arr123[:,0].astype(int)]
    else:
        ind_1_3x, ind_2_3x = [], []

    if len(coincidence_arr312):
        ind_3_3x = coincidence_arr312[:,0].astype(int)
    else:
        ind_3_3x = []

    mjd_1_3x = mjd_arr_1[ind_1_3x]
    mjd_2_3x = mjd_arr_2[ind_2_3x]
    mjd_3_3x = mjd_arr_3[ind_3_3x]

    coince_tup = [(data_1,ind_1_3x,mjd_1_3x),
                  (data_2,ind_2_3x,mjd_2_3x),
                  (data_3,ind_3_3x,mjd_3_3x)]

    # tuple of data frames for t<nday_lookback
    data_tup = (data_1, data_2, data_3)

    # Events that are coincident in time across 3 stations
    coince_tup_3x = (ind_1_3x, ind_2_3x, ind_3_3x)

    # Events that are coincident in time and DM across 2 stations
    coince_tup_2x = (ind_12, ind_21, ind_23, ind_32, ind_13, ind_31)

    return data_tup, coince_tup_3x, coince_tup_2x, first_MJD, last_MJD

def get_single_row(fncand, ind):
    data_ii = np.genfromtxt(fncand, skip_header=ind, max_rows=1)
    return data_ii

#def write_coincidences(coincidence_tup, fnout):
def write_coincidences(data_tup, coince_tup_3x, coince_tup_2x, fnout):
    data_1, data_2, data_3 = data_tup
    ind_1_3x, ind_2_3x, ind_3_3x = coince_tup_3x
    ind_12, ind_21, ind_23, ind_32, ind_13, ind_31 = coince_tup_2x

    try:
        data_1.insert(0, 'station', 1)
        data_2.insert(0, 'station', 2)
        data_3.insert(0, 'station', 3)
    except ValueError:
        pass 

    data_out = pd.DataFrame(data=None, index=None, columns=data_1.columns)
    data_out.astype(data_1.dtypes)
#    data_out.insert(0, 'station', 0)
    ncoinc_3x = len(ind_1_3x)

    data_1.index = range(len(data_1))
    data_2.index = range(len(data_2))
    data_3.index = range(len(data_3))

    # Add the 3 station temporal coincidences 
    # candidates for all three stations
    for ii in range(ncoinc_3x):
        data_out = data_out.append(data_1.loc[ind_1_3x[ii]])
        data_out = data_out.append(data_2.loc[ind_2_3x[ii]])
        data_out = data_out.append(data_3.loc[ind_3_3x[ii]])

    # Add the 2 station temporal/DM coincidences 
    # candidates for all three stations
    for ii in range(len(ind_12)):
        if ind_12[ii] in ind_1_3x:
            continue
        data_out = data_out.append(data_1.loc[ind_12[ii]])

        if ind_21[ii] in ind_2_3x:
            continue
        data_out = data_out.append(data_2.loc[ind_21[ii]])
            
    for ii in range(len(ind_23)):
        if ind_23[ii] in ind_2_3x or ind_23[ii] in ind_2_3x:
            continue
        data_out = data_out.append(data_2.loc[ind_23[ii]])
        if ind_32[ii] in ind_3_3x:
            continue
        data_out = data_out.append(data_3.loc[ind_32[ii]])

    for ii in range(len(ind_13)):
        if ind_13[ii] in ind_1_3x or ind_13[ii] in ind_12:
            continue
        data_out = data_out.append(data_1.loc[ind_13[ii]])
        if ind_31[ii] in ind_3_3x or ind_31[ii] in ind_32:
            continue
        data_out = data_out.append(data_3.loc[ind_31[ii]])

    data_out.index = range(len(data_out))
    data_out.astype(data_1.dtypes)
    data_out.to_csv(fnout)
    print("Saved to %s"%fnout)
    return data_out


def main(nday_lookback):
#    start_time = Time.now()
    rsync_heimdall_cand()
    fncand1 = '/home/user/cand_times_sync/heimdall.cand'
    fncand2 = '/home/user/cand_times_sync_od/heimdall_2.cand'
    fncand3 = '/home/user/cand_times_sync/heimdall_3.cand'

    data_tup,coince_tup_3x,coince_tup_2x,first_MJD,last_MJD = get_coincidence_3stations(
                                                                    fncand1, 
                                                                    fncand2, 
                                                                    fncand3, 
                                                                    t_thresh_sec=0.2, 
                                                                    nday_lookback=nday_lookback)

    x = Time(first_MJD, format='mjd')
    x = x.to_datetime()
    outdir = '/home/user/grex/%s%02d%02d%02d' % (str(x.year)[2:],x.month,x.day,x.hour)

    if os.path.isdir(outdir):
        pass
    else:
        os.system('mkdir %s' % outdir)

    if not len(coince_tup_3x)+len(coince_tup_2x):
        print("\nNo coincidences, exiting now.")
        os.system('touch %s/LastMJD%0.7f'%(outdir, last_MJD))
        exit()

    fnout = outdir + '/coincidence_3stations.csv'
    data_out = write_coincidences(data_tup, coince_tup_3x, 
                                  coince_tup_2x, fnout)

    os.system('touch %s/LastMJD%0.7f'%(outdir, last_MJD))

    return outdir

if __name__=='__main__':
    try:
        nday_lookback = float(sys.argv[1])
    except:
        nday_lookback = 1.

    outdir = main(nday_lookback)




