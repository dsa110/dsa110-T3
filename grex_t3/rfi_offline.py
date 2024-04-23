import sys

import numpy as np
import numpy.ma as ma
import pandas as pd

import analysis_tools

class RFI:
    """
    Class that holds a series of 
    RFI filters
    """

    def __init__(self, data, dumb_mask=[]):
        # Expectds filterbank data with shape (nfreq, ntime)
        self.data = data
        self.nfreq, self.ntime = data.shape 
        self.dumb_mask = dumb_mask
        self.dumb_mask_conjugate = [i for i in range(0,  self.nfreq) if i not in self.dumb_mask]

    def apply_dumb_mask(self):
        """ Mask out channels that 
        are always known to be bad
        """
        if len(self.dumb_mask):
            self.data.mask[self.dumb_mask, :] = True

    def remove_bandpass_Tsys(self):
        """ Remove bandpass based on system temperature
        """
        T_sys = np.mean(self.data.data, axis=1)
        bad_chans = T_sys < 0.001 * np.median(T_sys)
        T_sys[bad_chans] = 1
        self.data.data[:] /= T_sys[:,None]
        self.data.mask[bad_chans,:] = True

    def per_channel_sigmacut(self, frebin=1, sigma_thresh=3):
        """ Step through each channel and remove outlier time samples.
        """
        if frebin!=1:
            data_rb = self.data.reshape(self.nfreq//frebin, frebin, self.ntime)
            data_rb = data_rb.mean(1)
        else:
            data_rb = self.data

        for ii in range(data_rb.shape[0]):
            sig_ii = np.std(data_rb[ii])
            dmed_ii = np.median(data_rb[ii])
            bad_samp = np.where(data_rb[ii]>dmed_ii + sigma_thresh*sig_ii)[0]
            self.data[frebin*ii:frebin*(ii+1), bad_samp] = dmed_ii

    def per_channel_sigmacut_mproc(self, data, frebin=1, sigma_thresh=3):
        data_rb = data

        for ii in range(data_rb.shape[0]):
            sig_ii = np.std(data_rb[ii])
            dmed_ii = np.median(data_rb[ii])
            bad_samp = np.where(data_rb[ii]>dmed_ii + sigma_thresh*sig_ii)[0]
            data_rb[frebin*ii:frebin*(ii+1), bad_samp] = dmed_ii

        return data_rb

    def dm_zero_filter(self, sigma_thresh=7.0):
        """ Average over frequency to look 
        for DM=0 outliers in the timeseries. 
        """
        dmzero = np.mean(self.data,0)
        dmzero = dmzero - np.median(dmzero)
        s = pd.Series(dmzero)
        mad = np.mean(np.abs(s - s.mean()))
        stdev = 1.4826 * mad

        # Find DM=0 outliers 
        bad_samp = np.where(np.abs(dmzero) > sigma_thresh*stdev)[0]
        # data_replace = np.mean(self.data, 1).repeat(len(bad_samp))

        # # Replace bad samples with mean spectrum 
        # self.data[:, bad_samp] = data_replace.reshape(self.nfreq, 
        #                                               len(bad_samp))
        
        self.data.mask[:, bad_samp] = True

    def variancecut_freq(self, axis=1, sigma_thresh=3):
        """ Cut on variance outliers along specified
        axis 
        """
        sig = np.std(self.data, axis=axis)
        sigsig = np.std(sig)
        meansig = np.median(sig)
        ind = np.where(sig > meansig + sigma_thresh*sigsig)[0]
        if axis==0:
            self.data.mask[:, ind] = True
        elif axis==1:
            self.data.mask[ind] = True

    def variancecut_time(self, axis=0, sigma_thresh=3):
        """ Cut on variance outliers along specified
        axis 
        """
        sig = np.std(self.data.data, axis=axis)
        sigsig = np.std(sig)
        meansig = np.mean(sig)
        ind = np.where(sig > meansig + sigma_thresh*sigsig)[0]
        if axis==0:
            self.data.mask[:, ind] = True
        elif axis==1:
            self.data.mask[ind] = True

    def detrend_data(self,axis=0,degree=4):
        """ Detrend data along specified axis
        with a polynomial of given degree
        """
        if axis==1:
            xval = np.arange(self.ntime)
            meanaxis=0
        elif axis==0:
            xval = np.arange(self.nfreq)
            meanaxis=1
        else:
            assert "Expected axis=0 or axis=1"

        p = np.polyfit(xval, np.mean(self.data.data,axis=meanaxis), 4)
        f = np.poly1d(p) 

        if axis==1:
            self.data.data[:] -= f(xval)[None]
        elif axis==0:
            self.data.data[:] -= f(xval)[:,None]

def apply_rfi_filters_grex(data, sigma_thresh_chan=3.,
                           sigma_thresh_dm0=7., perchannel=False,
                           dumb_mask=[]):
    """ Apply in sequence RFI filters and detrending 
    to time frequency intensity data. 

    Parameters:
    ----------
    data: ndarray
        (nfreq, ntime)
    sigma_thresh_chan : float 
        threshold sigma for per channel excision 
    sigma_thesh_dm0 : float 
        threshold sigma in DM=0 timeseries 
    perchannel : bool 
        clip on a per-channel timeseries basis. very slow, be warned.
    dumb_mask : list
        list of channels to ignore 

    Returns:
    -------
    R.data : ndarray
        RFI cleaned data
    """
    mask = np.zeros_like(data, dtype=bool)
    datamask = ma.masked_array(data.copy(), mask=mask)
    R = RFI(datamask, dumb_mask=dumb_mask)
    R.apply_dumb_mask()
    R.remove_bandpass_Tsys()

    if perchannel:
        R.per_channel_sigmacut(1, sigma_thresh_chan)

    # Apply a cut on variance in time domain
    R.variancecut_time(axis=0, sigma_thresh=3)
    # Apply a cut on variance for frequency spectrum
    R.variancecut_freq(axis=1, sigma_thresh=3)
    # Apply a cut on variance in time domain
    R.variancecut_time(axis=0, sigma_thresh=5)
    # Apply a cut on variance for frequency spectrum
    R.variancecut_freq(axis=1, sigma_thresh=5)
    # Sum over the frequency axis and cut on the DM=0 timeseries 
    R.dm_zero_filter(sigma_thresh_dm0)
    # Detrend time series with degree 4 polynomial 
    R.detrend_data(axis=0,degree=4)
    # Detrend spectrum with degree 4 polynomial 
    R.detrend_data(axis=1,degree=4)
    
    return R.data 

if __name__=='__main__':
    fn_fil = sys.argv[1]
    fn_out_fil = sys.argv[2]
    chunksize = 2**12
    
    for ii in range(int(1e8)):
        data_fil_obj, freq_arr, dt, header = analysis_tools.read_fil_data_grex(fn_fil, 
                                            start=ii*chunksize, stop=chunksize)
        if data_fil_obj.data.shape[1]==0:
            break

        data = apply_rfi_filters_grex(data_fil_obj.data)
        print(f"Done cleaning chunk {ii}")
        continue
        if ii==0:
            reader.write_to_fil(np.zeros([header['nchans'], 0]), header, fn_out_fil)

        fil_obj = reader.filterbank.FilterbankFile(fn_out_fil, mode='readwrite')
        fil_obj.append_spectra(data.transpose())

    if data_fil_obj.data.shape[1]!=0:
        print("Did not reach end of file, maybe crank up loop range")
