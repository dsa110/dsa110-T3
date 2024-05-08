import os

import time
import numpy as np 
import pandas as pd
from astropy.time import Time
from astropy import units as u
import scipy

sgr_1935_ra = 293.731999*u.deg
dm_sgr1935 = 332.

station_id = {1:'OVRO', 2:'Delta', 3:'Goldstone'}

def get_RA_overhead(mjd=None,longitude_rad=-2.064427799136453):
    """ Calculate RA from MJD assuming OVRO longitude
    """
    if mjd is None:
        t = Time.now()
    elif mjd > 40000:
        t = Time(mjd, format='mjd')
    else:
        print("Expecting MJD as float >500000 or None")
        return

    t.delta_ut1_utc = 0.
    ra = t.sidereal_time('mean', longitude_rad*u.rad)
    return ra

def get_RA_diff(mjd, src_RA):
    """ Get difference in RA between source and hour angle
    """
    ra = get_RA_overhead(mjd)
    return ra-src_RA

def get_mjd_cand(fncand):
    """ Open candidate metadata file fncand and 
    create array of MJDs
    """
    f = open(fncand,'r')

    mjd_arr = []
    for line in f:
        ll = line.split(' ')
        try:
            mjd = np.float(ll[8])+(np.float(ll[9])+np.float(ll[10])/60.+np.float(ll[11])/3600.0)/24.0
        except:
            mjd = -1
        mjd_arr.append(mjd)

    return np.array(mjd_arr)

def get_mjd_cand_pd(data, offset_utc_hours=7):
    """ Get MJD array from pandas dataframe  
    """
    mjd = data['mjd_day']+(data['mjd_hr']+offset_utc_hours+data['mjd_min']/60.+data['mjd_sec']/3600.)/24.

    return mjd

def read_heim_pandas(fncand, skiprows=0):
    """ Read in Heimdall candidates txt file and create
        pandas dataframe 
    """
    if fncand.endswith('csv'):
        data = pd.read_csv(fncand)
        return data
    elif fncand.endswith('cand'):

        data = pd.read_table(fncand,
                      skiprows=skiprows, delimiter=' ', 
                      header=None, usecols=range(12))

        heim_columns = ['snr','cand','time_sec',
                    'log2width','unknown2','dm',
                    'unknown3','mjdx','mjd_day',
                    'mjd_hr','mjd_min','mjd_sec']

        data.columns = heim_columns
        return data
    else:
        print("Expected a .csv or .cand file.")
        return 

def rsync_heimdall_cand():
    os.system("rsync -avzhe ssh user@158.154.14.10:/home/user/cand_times_sync/ /home/user/cand_times_sync");
    os.system("rsync -avzhe ssh user@166.140.120.248:/home/user/cand_times_sync/ /home/user/cand_times_sync_od");
    time.sleep(60)

def check_latest_cand_time():
    flist = glob.glob('/home/user/candidates/*.fil')
    flmax = max(flist, key=os.path.getctime)
    time_since_cand = os.path.getctime(flmax)-time.time()
    return time_since_cand

def read_fil_data_grex(fn, start=0, stop=1):
    """ Read in filterbank data"""
    fil_obj = FilReader(fn)
    header = fil_obj.header
    delta_t = header.tsamp # delta_t in seconds                                
    fch1 = header.fch1
    nchans = header.nchans
    foff = header.foff
    fch_f = fch1 + nchans*foff
    freq = np.linspace(fch1,fch_f,nchans)
    try:
        data = fil_obj.read_block(start, stop)
    except(ValueError):
        data = fil_obj.read_block(start, header.nsamples)
    return data, freq, delta_t, header



class SNR_Tools:

    def __init__(self):
        pass

    def sigma_from_mad(self, data):
        """ Get gaussian std from median 
        aboslute deviation (MAD)
        """
        assert len(data.shape)==1, 'data should be one dimensional'

        med = np.median(data)
        mad = np.median(np.absolute(data - med))

        return 1.4826*mad, med

    def calc_snr_presto(self, data, verbose=None):
        """ Calculate S/N of 1D input array (data)
        after excluding 0.05 at tails
        """
        std_chunk = scipy.signal.detrend(data, type='linear')
        std_chunk.sort()
        ntime_r = len(std_chunk)
        stds = 1.148*np.sqrt((std_chunk[ntime_r//40:-ntime_r//40]**2.0).sum() /
                              (0.95*ntime_r))
        snr_ = std_chunk[-1] / stds 
        if verbose==None:
            return snr_
        else:
            return snr_, stds

    def calc_snr_amber(self, data, thresh=3.):
        sig = np.std(data)
        dmax = (data.copy()).max()
        dmed = np.median(data)
        N = len(data)

        # remove outliers 4 times until there 
        # are no events above threshold*sigma
        for ii in range(4):
            ind = np.where(np.abs(data-dmed)<thresh*sig)[0]
            sig = np.std(data[ind])
            dmed = np.median(data[ind])
            data = data[ind]
            N = len(data)

        snr_ = (dmax - dmed)/(1.048*sig)

        return snr_

    def calc_snr_mad(self, data):
        sig, med = self.sigma_from_mad(data)

        return (data.max() - med) / sig

    def calc_snr_matchedfilter(self, data, widths=None, true_filter=None):
        """ Calculate the S/N of pulse profile after 
        trying 9 rebinnings.
        Parameters
        ----------
        arr   : np.array
            (ntime,) vector of pulse profile 
        Returns
        -------
        snr : np.float 
            S/N of pulse
        """
        assert len(data.shape)==1
        
        ntime = len(data)
        snr_max = 0
        width_max = 0
#        data = scipy.signal.detrend(data, type='linear')

        if widths is None:
            widths = [1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500]

        for ii in widths:
            if true_filter is None:
                mf = np.ones([ii])
            else:
                mf = true_filter
            data_mf = scipy.correlate(data, mf)
            snr_ = self.calc_snr_amber(data_mf)

            if snr_ > snr_max:
                snr_max = snr_
                width_max = ii

        return snr_max, width_max
        

    def calc_snr_widths(self, data, widths=None):
        """ Calculate the S/N of pulse profile after 
        trying 9 rebinnings.
        Parameters
        ----------
        arr   : np.array
            (ntime,) vector of pulse profile 
        Returns
        -------
        snr : np.float 
            S/N of pulse
        """
        assert len(data.shape)==1
        
        ntime = len(data)
        snr_max = 0
        data -= np.median(data)

        if widths is None:
            widths = [1, 2, 4, 8, 16, 32, 64, 128]

    #    for ii in range(1, 10):
        for ii in widths:
            for jj in range(ii):
                # skip if boxcar width is greater than 1/4th ntime
                if ii > ntime//8:
                    continue
                
                arr_copy = data.copy()
                arr_copy = np.roll(arr_copy, jj)
                arr_ = arr_copy[:ntime//ii*ii].reshape(-1, ii).mean(-1)

                snr_ = self.calc_snr(arr_)

                if snr_ > snr_max:
                    snr_max = snr_
                    width_max = ii

        return snr_max, width_max

    def compare_snr(self, fn_1, fn_2, dm_min=0, dm_max=np.inf, save_data=False,
                    sig_thresh=5.0, t_window=0.5, max_rows=None,
                    t_max=np.inf, tab=None, freq_ref_1=1400., freq_ref_2=1400.):
        """ Read in two files with single-pulse candidates
        and compare triggers.
        Parameters:
        ----------
        fn_1 : str 
            name of input triggers text file
            (must be .trigger, .singlepulse, or .txt)
        fn_2 : str
            name of input triggers text file for comparison 
        dm_min : float
            do not process triggers below this DM 
        dm_max : float 
            do not process triggers above this DM 
        save_data : bool 
            if True save to np.array
        sig_thresh : float 
            do not process triggers below this S/N 
        t_window : float 
            time window within which triggers in 
            fn_1 and fn_2 will be considered the same 
        Return:
        -------
        Function returns four parameter arrays for 
        each fn_1 and fn_2, which should be ordered so 
        that they can be compared directly:
        grouped_params1, grouped_params2, matched_params
        """
        snr_1, dm_1, t_1, w_1, ind_full_1 = get_triggers(fn_1, sig_thresh=sig_thresh, 
                                    dm_min=dm_min, dm_max=dm_max, t_window=t_window, 
                                                         max_rows=max_rows, t_max=t_max, tab=tab)

        snr_2, dm_2, t_2, w_2, ind_full_2 = get_triggers(fn_2, sig_thresh=sig_thresh, 
                                    dm_min=dm_min, dm_max=dm_max, t_window=t_window, 
                                                         max_rows=max_rows, t_max=t_max, tab=tab)

        # adjust arrival times to have same ref freq after dedispersion
        t_1 += 4148*dm_1*(freq_ref_2**-2 - freq_ref_1**-2)

        snr_2_reorder = []
        dm_2_reorder = []
        t_2_reorder = []
        w_2_reorder = []

        ntrig_1 = len(snr_1)
        ntrig_2 = len(snr_2)    

        par_1 = np.concatenate([snr_1, dm_1, t_1, w_1, ind_full_1]).reshape(5, -1)
        par_2 = np.concatenate([snr_2, dm_2, t_2, w_2, ind_full_2]).reshape(5, -1)

        # Make arrays for the matching parameters
        par_match_arr = []
        ind_missed = []
        ind_matched = []

        for ii in range(len(snr_1)):

            tdiff = np.abs(t_1[ii] - t_2)
            ind = np.where(tdiff == tdiff.min())[0]

            if t_1[ii] > t_max:
                continue

            # make sure you are getting correct trigger in dm/time space
            if len(ind) > 1:
                ind = ind[np.argmin(np.abs(dm_1[ii]-dm_2[ind]))]
            else:
                ind = ind[0]

            # check for triggers that are within 1.0 seconds and 20% in dm
            if (tdiff[ind]<1.0) and (np.abs(dm_1[ii]-dm_2[ind])/dm_1[ii])<0.2:
                pparams = (tdiff[ind], t_1[ii], t_2[ind], dm_1[ii], dm_2[ind], snr_1[ii], snr_2[ind], w_1[ii], w_2[ind])
                print("%1.4f  %5.1f  %5.1f  %5.1f  %5.1f %5.1f  %5.1f %5.1f  %5.1f" % pparams)

                params_match = np.array([snr_1[ii], snr_2[ind], 
                                         dm_1[ii], dm_2[ind],
                                         t_1[ii], t_2[ind],
                                         w_1[ii], w_2[ind]])

                par_match_arr.append(params_match)
                ind_matched.append(ii)

            else:
                # Keep track of missed triggers
                ind_missed.append(ii)

        if len(par_match_arr)==0:
            print("No matches found")
            return 

        # concatenate list and reshape to (nparam, nmatch, 2 files)
        par_match_arr = np.concatenate(par_match_arr).reshape(-1, 4, 2)
        par_match_arr = par_match_arr.transpose((1, 0, 2))

        if save_data is True:
            nsnr = min(len(snr_1), len(snr_2))
            snr_1 = snr_1[:nsnr]
            snr_2 = snr_2_reorder[:nsnr]

            np.save(fn_1+'_params_grouped', par_1)
            np.save(fn_2+'_params_grouped', par_2)
            np.save('params_matched', par_match_1)

        return par_1, par_2, par_match_arr, ind_missed, ind_matched  

    def plot_comparison(self, par_1, par_2, par_match_arr, ind_missed, figname='./test.pdf'):
        fig = plt.figure(figsize=(14,14))

        frac_recovered = len(ind_missed)

        snr_1, snr_2 = par_1[0], par_2[0]
        dm_1, dm_2 = par_1[1], par_2[1]
        width_1, width_2 = par_1[3], par_2[3]

        snr_1_match = par_match_arr[0,:,0]
        snr_2_match = par_match_arr[0,:,1]

        dm_1_match = par_match_arr[1,:,0]
        dm_2_match = par_match_arr[1,:,1]

        width_1_match = par_match_arr[3,:,0]
        width_2_match = par_match_arr[3,:,1]

        fig.add_subplot(311)
        plt.plot(snr_1_match, snr_2_match, '.')
        plt.plot(snr_1, snr_1, color='k')
        plt.plot(snr_1[ind_missed], np.zeros([len(ind_missed)]), 'o', color='orange')
        plt.xlabel('Injected S/N', fontsize=13)
        plt.ylabel('Detected S/N', fontsize=13)        
        plt.legend(['Detected events','Expected S/N','Missed events'], fontsize=13)

        fig.add_subplot(312)
        plt.plot(dm_1_match, snr_1_match/snr_2_match, '.')
        plt.plot(dm_1[ind_missed], np.zeros([len(ind_missed)]), 'o', color='orange')
        plt.xlabel('DM', fontsize=13)
        plt.ylabel('Expected S/N : Detected S/N', fontsize=13)        
        plt.legend(['Detected events','Missed events'], fontsize=13)

        fig.add_subplot(337)
        plt.hist(width_1, bins=50, alpha=0.3, normed=True)
        plt.hist(width_2, bins=50, alpha=0.3, normed=True)
        plt.hist(width_1[ind_missed], bins=50, alpha=0.3, normed=True)
        plt.xlabel('Width [samples]', fontsize=13)

        fig.add_subplot(338)
        plt.plot(width_1_match, snr_1_match,'.')
        plt.plot(width_1_match, snr_2_match,'.')
        plt.plot(width_1, snr_1, '.')
        plt.xlabel('Width [samples]', fontsize=13)
        plt.ylabel('S/N injected', fontsize=13)

        fig.add_subplot(339)
        plt.plot(width_1_match, dm_1_match,'.')
        plt.plot(width_1_match, dm_2_match,'.')
        plt.plot(width_1, dm_1,'.')
        plt.xlabel('Width [samples]', fontsize=13)
        plt.ylabel('DM', fontsize=13)

        plt.tight_layout()
        plt.show()
        plt.savefig(figname)

def read_voltage(fn):
    f = h5py.File(fn,'r')

    # Should be (ntime, npol, nchan, {re/img})
    data = f['voltages'][:]
    return data 

def voltage_to_stokes(data):
    x = np.sqrt(data[:, 0, :, 0]**2 + data[:, 0, :, 1]**2)
    y = np.sqrt(data[:, 1, :, 0]**2 + data[:, 1, :, 1]**2)

    StokesI = x**2 + y**2

    return StokesI



# @jit
# def shift_data(data, delay):
#     """
#     Shift data by a certain number of samples.
    
#     Parameters
#     ----------
#     data : array
#         The input data, with shape (ntime,).
#     delay : int
#         The number of samples to shift the data.

#     Returns
#     -------
#     array
#         The shifted data, with the same shape as the input data.
#     """
#     return np.roll(data, delay)
