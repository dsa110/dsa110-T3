# plot triggered FRB candidates
# liam.dean.connor@gmail.com & ghellbourg@astro.caltech.edu
# 25/02/2021

import os
import os.path
import sys

import scipy.signal
from scipy import stats
import numpy as np
import matplotlib as mpl
mpl.rcdefaults()
mpl.use('Agg') # hack
import matplotlib.pyplot as plt 
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import json
import glob
import optparse
import pandas
import h5py

from sigpyproc.Readers import FilReader
import slack_sdk as slack
import astropy.units as u
from astropy.time import Time
import dsautils.coordinates
import dsautils.dsa_store as ds

MLMODELPATH='/home/ubuntu/connor/MLmodel/20190501freq_time.hdf5' # Keras neural network model for Freq/Time array
webPLOTDIR='/dataz/dsa110/operations/T3/'
T1dir = '/dataz/dsa110/operations/T1'
T2dir = '/dataz/dsa110/operations/T2/cluster_output'

d = ds.DsaStore()

# set up slack client
slack_file = '{0}/.config/slack_api'.format(
    os.path.expanduser("~")
)
if not os.path.exists(slack_file):
    raise RuntimeError(
        "Could not find file with slack api token at {0}".format(
            slack_file
        )
    )
with open(slack_file) as sf_handler:
    slack_token = sf_handler.read()
    slack_client = slack.WebClient(token=slack_token)


plt.rcParams.update({
                    'font.size': 12,
                    'font.family': 'serif',
                    'axes.labelsize': 14,
                    'axes.titlesize': 15,
                    'xtick.labelsize': 12,
                    'ytick.labelsize': 12,
                    'xtick.direction': 'in',
                    'ytick.direction': 'in',
                    'xtick.top': True,
                    'ytick.right': True,
                    'lines.linewidth': 0.5,
                    'lines.markersize': 5,
                    'legend.fontsize': 14,
                    'legend.borderaxespad': 0,
                    'legend.frameon': False,
                    'legend.loc': 'lower right'})


def read_fil_data_dsa(fn, start=0, stop=1):
    """ Read in filterbank data
    """
    fil_obj = FilReader(fn)
    header = fil_obj.header
    delta_t = fil_obj.header['tsamp'] # delta_t in seconds                                                                                                                  
    fch1 = header['fch1']
    nchans = header['nchans']
    foff = header['foff']
    fch_f = fch1 + nchans*foff
    freq = np.linspace(fch1,fch_f,nchans)
    try:
        data = fil_obj.readBlock(start, stop)
    except(ValueError):
        data = 0

    return data, freq, delta_t, header


def plotfour(dataft, datats, datadmt, 
             beam_time_arr=None, figname=None, dm=0,
             dms=[0,1], 
             datadm0=None, suptitle='', heimsnr=-1,
             ibox=1, ibeam=-1, prob=-1,
             showplot=True,multibeam_dm0ts=None,
             fnT2clust=None,imjd=0.0,fake=False):
    """ Plot a trigger's dynamics spectrum, 
        dm/time array, pulse profile, 
        multibeam info (optional), and zerodm (optional)

        Parameter
        ---------
        dataft : 
            freq/time array (nfreq, ntime)
        datats : 
            dedispersed timestream
        datadmt : 
            dm/time array (ndm, ntime)
        beam_time_arr : 
            beam time SNR array (nbeam, ntime)
        figname : 
            save figure with this file name 
        dm : 
            dispersion measure of trigger 
        dms : 
            min and max dm for dm/time array 
        datadm0 : 
            raw data timestream without dedispersion
    """

    classification_dict = {'prob' : [],
                           'snr_dm0_ibeam' : [],
                           'snr_dm0_allbeam' : []}
    datats /= np.std(datats[datats!=np.max(datats)])
    nfreq, ntime = dataft.shape
    xminplot,xmaxplot = 500.-300*ibox/16.,500.+300*ibox/16 # milliseconds
    if xminplot<0:
        xmaxplot=xminplot+500+300*ibox/16        
        xminplot=0
#    xminplot,xmaxplot = 0, 1000.
    dm_min, dm_max = dms[0], dms[1]
    tmin, tmax = 0., 1e3*dataft.header['tsamp']*ntime
    freqmax = dataft.header['fch1']
    freqmin = freqmax + dataft.header['nchans']*dataft.header['foff']
    freqs = np.linspace(freqmin, freqmax, nfreq)
    tarr = np.linspace(tmin, tmax, ntime)
    fig, axs = plt.subplots(3, 2, figsize=(8,10), constrained_layout=True)

    if fake:
        fig.patch.set_facecolor('red')
        fig.patch.set_alpha(0.5)

    extentft=[tmin,tmax,freqmin,freqmax]
    axs[0][0].imshow(dataft, aspect='auto',extent=extentft, interpolation='nearest')
    DM0_delays = xminplot + dm * 4.15E6 * (freqmin**-2 - freqs**-2)
    axs[0][0].plot(DM0_delays, freqs, c='r', lw='2', alpha=0.35)
    axs[0][0].set_xlim(xminplot,xmaxplot)
    axs[0][0].set_xlabel('Time (ms)')
    axs[0][0].set_ylabel('Freq (MHz)')
    if prob!=-1:
        axs[0][0].text(xminplot+50*ibox/16.,0.5*(freqmax+freqmin),
                       "Prob=%0.2f" % prob, color='white', fontweight='bold')
        classification_dict['prob'] = prob

#    plt.subplot(322)
    extentdm=[tmin, tmax, dm_min, dm_max]
    axs[0][1].imshow(datadmt[::-1], aspect='auto',extent=extentdm)
    axs[0][1].set_xlim(xminplot,xmaxplot)
    axs[0][1].set_xlabel('Time (ms)')
    axs[0][1].set_ylabel(r'DM (pc cm$^{-3}$)')

#    plt.subplot(323)
    axs[1][0].plot(tarr, datats)
    axs[1][0].grid('on', alpha=0.25)
    axs[1][0].set_xlabel('Time (ms)')
    axs[1][0].set_ylabel(r'Power ($\sigma$)')
    axs[1][0].set_xlim(xminplot,xmaxplot)
    axs[1][0].text(0.51*(xminplot+xmaxplot), 0.5*(max(datats)+np.median(datats)), 
            'Heimdall S/N : %0.1f\nHeimdall DM : %d\
            \nHeimdall ibox : %d\nibeam : %d' % (heimsnr,dm,ibox,ibeam), 
            fontsize=8, verticalalignment='center')
    
#    parent_axes=fig.add_subplot(324)
    parent_axes = axs[1][1]
    if beam_time_arr is None:
        plt.xticks([])
        plt.yticks([])
        plt.text(0.20, 0.55, 'Multibeam info\n not available',
                fontweight='bold')
    else:
        parent_axes.imshow(beam_time_arr[::-1], aspect='auto', extent=[tmin, tmax, 0, beam_time_arr.shape[0]], 
                  interpolation='nearest')
        parent_axes.axvline(540, ymin=0, ymax=6, color='r', linestyle='--', alpha=0.55)
        parent_axes.axvline(460, ymin=0, ymax=6, color='r', linestyle='--', alpha=0.55)
        parent_axes.axhline(max(0,ibeam-1), xmin=0, xmax=100, color='r', linestyle='--', alpha=0.55)
        parent_axes.axhline(ibeam+3, xmin=0, xmax=100, color='r', linestyle='--', alpha=0.55)
        parent_axes.set_xlim(xminplot,xmaxplot)
        parent_axes.set_xlabel('Time (ms)')
        parent_axes.set_ylabel('Beam', fontsize=15)
        small_axes = inset_axes(parent_axes,
                                width="25%", # width = 30% of parent_bbox
                                height="25%", # height : 1 inch
                                loc=4)
        small_axes.imshow(beam_time_arr[::-1][ibeam-4:ibeam+4],
                          aspect='auto',
                          extent=[tmin, tmax, ibeam-4, ibeam+4],
                          interpolation='nearest', cmap='afmhot')
        small_axes.set_xlim(400., 600.)

    if datadm0 is not None:
#        plt.subplot(325)
        datadm0 -= np.median(datadm0.mean(0))
        datadm0_sigmas = datadm0.mean(0)/np.std(datadm0.mean(0)[-500:])
        snr_dm0ts_iBeam = np.max(datadm0_sigmas)
        axs[2][0].plot(np.linspace(0, tmax, len(datadm0[0])), datadm0_sigmas, c='k')
        classification_dict['snr_dm0_ibeam'] = snr_dm0ts_iBeam
        
        if multibeam_dm0ts is not None:
#        if False:
            multibeam_dm0ts -= np.median(multibeam_dm0ts)            
#            multibeam_dm0ts = multibeam_dm0ts/np.std(multibeam_dm0ts[multibeam_dm0ts!=multibeam_dm0ts.max()])
            multibeam_dm0ts = multibeam_dm0ts/np.std(multibeam_dm0ts[-500:])
            snr_dm0ts_allbeams = np.max(multibeam_dm0ts)
            axs[2][0].plot(np.linspace(0, tmax, len(multibeam_dm0ts)), multibeam_dm0ts, color='C1', alpha=0.75)
            axs[2][0].legend(['iBeam=%d'%ibeam, 'All beams'], loc=1, fontsize=10)
            axs[2][0].set_ylabel(r'Power ($\sigma$)')
            classification_dict['snr_dm0_allbeam'] = snr_dm0ts_allbeams
        else:
            axs[2][0].legend(['DM=0 Timestream'], loc=2, fontsize=10)
        axs[2][0].set_xlabel('Time (ms)')
                
        if fnT2clust is not None:
            T2object = pandas.read_csv(fnT2clust)
            ind = np.where(np.abs(86400*(imjd-T2object.mjds[:]))<30.0)[0]
            ttsec = (T2object.mjds.values-imjd)*86400
            mappable = axs[2][1].scatter(ttsec[ind],
                                         T2object.ibeam[ind],
                                         c=T2object.dm[ind],
                                         s=2*T2object.snr[ind],
                                         cmap='RdBu_r',
                                         vmin=0,vmax=1200)
            fig.colorbar(mappable, label=r'DM (pc cm$^{-3}$)', ax=axs[2][1])
            axs[2][1].scatter(0, ibeam, s=100, marker='s',
                        facecolor='none', edgecolor='black')
            axs[2][1].set_xlim(-10,10.)
            axs[2][1].set_ylim(0,256)
            axs[2][1].set_xlabel('Time (s)')
            axs[2][1].set_ylabel('ibeam')

    not_real = False

    if multibeam_dm0ts is not None:
        if classification_dict['snr_dm0_allbeam']>7.5:
            if classification_dict['snr_dm0_ibeam']>10.:
                if classification_dict['prob']<0.25:
                    not_real = True

    try:
        if classification_dict['prob']<0.01:
            not_real = True
    except:
        pass

    if not_real==True:
        suptitle += ' (Probably not real)'

    fig.suptitle(suptitle, color='C1')
    if fake:
        fig.suptitle('INJECTION')

    if figname is not None:
        fig.savefig(figname)
    if showplot:
        fig.show()
    else:
        plt.close(fig)

    return not_real
        

def dm_transform(data, dm_max=20,
                 dm_min=0, dm0=None, ndm=64, 
                 freq_ref=None, downsample=16):
    """ Transform freq/time data to dm/time data.                                                                                                """

    ntime = data.shape[1]

    dms = np.linspace(dm_min, dm_max, ndm, endpoint=True)

    if dm0 is not None:
        dm_max_jj = np.argmin(abs(dms-dm0))
        dms += (dm0-dms[dm_max_jj])

    data_full = np.zeros([ndm, ntime//downsample])

    for ii, dm in enumerate(dms):
        dd = data.dedisperse(dm)
        _dts = np.mean(dd,axis=0)
        data_full[ii] = _dts[:ntime//downsample*downsample].reshape(ntime//downsample, downsample).mean(1)

    return data_full, dms


def proc_cand_fil(fnfil, dm, ibox, snrheim=-1, 
                  pre_rebin=1, nfreq_plot=64,
                  heim_raw_tres=1, 
                  rficlean=False, ndm=64,
                  norm=True, freq_ref=None):
    """ Take filterbank file path, preprocess, and 
    plot trigger

    Parameters:
    ----------

    fnfil   : str 
        path to .fil file 
    DM      : float 
        dispersion measure of trigger 
    ibox    : int 
        preferred boxcar width 
    snrheim : float 
        S/N of candidate found by Heimdall
    pre_rebin : int 
        rebin in time by this factor *before* dedispersion (saves time)
    nfreq_plot : int 
        number of frequency channels in output
    heim_raw_tres : 32  
    """
    header = read_fil_data_dsa(fnfil, 0, 1)[-1]
    # read in 4 seconds of data
    nsamp = int(4.0/header['tsamp'])
    data, freq, delta_t_raw, header = read_fil_data_dsa(fnfil, start=0, 
                                                       stop=nsamp)

    nfreq0, ntime0 = data.shape

    if pre_rebin>1:
        # Ensure that you do not pre-downsample by more than the total boxcar
        pre_rebin = min(pre_rebin, ibox*heim_raw_tres)
        data = data.downsample(pre_rebin)

    datadm0 = data.copy()
    
    if rficlean:
        data = cleandata(data, clean_type='aladsa')

    tsdm0 = np.mean(data,axis=0)

    dm_err = ibox / 1.0 * 25.
    dm_err = 250.0
    datadm, dms = dm_transform(data, dm_max=dm+dm_err,
                               dm_min=dm-dm_err, dm0=dm, ndm=ndm, 
                               freq_ref=freq_ref, 
                               downsample=heim_raw_tres*ibox//pre_rebin)
    data = data.dedisperse(dm)
    data = data.downsample(heim_raw_tres*ibox//pre_rebin)
    data = data.reshape(nfreq_plot, data.shape[0]//nfreq_plot, 
                        data.shape[1]).mean(1)

    if norm:
        data = data-np.median(data,axis=1,keepdims=True)
        data /= np.std(data)

    return data, datadm, tsdm0, dms, datadm0


def medflagdata(spec, filtsize, thres):
    specfilt = scipy.signal.medfilt(spec,kernel_size=int(filtsize));
    speccorrec = spec - specfilt;
    specstd = stats.median_absolute_deviation(speccorrec);
    return np.concatenate((np.argwhere(speccorrec > thres*specstd),np.argwhere(speccorrec < -thres*specstd)))


def cleandata(data, threshold_time=3.25, threshold_frequency=2.75, bin_size=32,
              n_iter_time=3, n_iter_frequency=3, clean_type='time', wideclean=None):
    """ Take filterbank object and mask
    RFI time samples with average spectrum.

    Parameters:
    ----------
    data :
        data array (nfreq, ntime)
    threshold_time : float
        units of sigma
    threshold_frequency : float
        units of sigma
    bin_size : int
        quantization bin size
    n_iter_time : int
        Number of iteration for time cleaning
    n_iter_frequency : int
        Number of iteration for frequency cleaning
    clean_type : str
        type of cleaning to be done.
        Accepted values: 'time', 'frequency', 'both', 'perchannel'

    Returns:
    -------
    cleaned filterbank object
    """
    if clean_type not in ['time', 'both', 'frequency', 'perchannel', 'aladsa']:
        return data
        
    nfreq = data.shape[0]
    ntimes = data.shape[1]

    dtmean = np.mean(data, axis=-1)
    # Clean in time
    #sys_temperature_bandpass(data.data)
    #remove_noisy_freq(data.data, 3)
    #remove_noisy_channels(data.data, sigma_threshold=2, iters=5)
    if clean_type in ['time', 'both']:
        for i in range(n_iter_time):
            dfmean = np.mean(data, axis=0)
            stdevf = np.std(dfmean)
            medf = np.median(dfmean)
            maskf = np.where(np.abs(dfmean - medf) > threshold_time*stdevf)[0]
            # replace with mean spectrum
            data[:, maskf] = dtmean[:, None]*np.ones(len(maskf))[None]
            
    if clean_type=='aladsa':
        print('flagging a la DSA\n');
        meanidx = medflagdata(dtmean, 21, 5.);
        varidx = medflagdata(np.var(data,axis=-1), 21, 5.);
        allidx = np.concatenate((meanidx,varidx));
        allidx = np.asarray(list(set(list(np.ravel(allidx)))));
        data[allidx,:] = np.zeros((len(allidx),ntimes));
        

    if clean_type=='perchannel':
        for ii in range(n_iter_time):
            dtmean = np.mean(data, axis=1, keepdims=True)
            dtsig = np.std(data, axis=1)
            for nu in range(data.shape[0]):
                d = dtmean[nu]
                sig = dtsig[nu]
                maskpc = np.where(np.abs(data[nu]-d)>threshold_time*sig)[0]
                data[nu][maskpc] = d

    # Clean in frequency
    # remove bandpass by averaging over bin_size ajdacent channels
    if clean_type in ['frequency', 'both']:
        for ii in range(n_iter_frequency):
            dtmean_nobandpass = data.mean(1) - dtmean.reshape(-1, bin_size).mean(-1).repeat(bin_size)
            stdevt = np.std(dtmean_nobandpass)
            medt = np.median(dtmean_nobandpass)
            maskt = np.abs(dtmean_nobandpass - medt) > threshold_frequency*stdevt
            data[maskt] = np.median(dtmean)

    return data


def generate_beam_time_arr(fl, ibeam=0, pre_rebin=1, 
                           dm=0, ibox=1, heim_raw_tres=1):
    """ Take list of nbeam .fil files, dedisperse each 
    to the dm of the main trigger, and generate an 
    (nbeam, ntime) SNR array.

    Parameters:
    -----------
    fl : list 
        list of .fil files, each 4 seconds long
    ibeam : int 
        beam number of trigger
    pre_rebin : 
        downsample by this factor before dedispersion to save time
    dm : int 
        dm of ibeam candidate
    ibox : int 
        boxcar width of ibeam candidate 
    heim_raw_tres : int 
        ratio of 

    Returns:
    --------
    beam_time_arr : ndarray 
        array of SNR values (nbeam, ntime)
    """
    fl.sort()
    nbeam = len(fl[:])
    header = read_fil_data_dsa(fl[0], 0, 1)[-1]
    # read in 4 seconds of data
    nsamp = int(4.0/header['tsamp'])
    nsamp_final = nsamp // (heim_raw_tres*ibox)
    nfreq_final = 1024
#    beam_time_arr = np.zeros([nbeam, nsamp_final])
    beam_time_arr = np.zeros([nbeam, nfreq_final, nsamp_final])    
    multibeam_dm0ts = 0
    beamno_arr=[]
    
    for jj,fnfil in enumerate(fl):
        print(fnfil, beam_time_arr.shape)
        beamno = int(fnfil.strip('.fil').split('_')[-1])
        data, freq, delta_t_raw, header = read_fil_data_dsa(fnfil, start=0, 
                                                           stop=nsamp)
        nfreq0, ntime0 = data.shape

        # Ensure that you do not pre-downsample by more than the total boxcar
        pre_rebin = min(pre_rebin, ibox*heim_raw_tres)

        multibeam_dm0ts += data.mean(0) 
        # Rebin in frequency by 8x
        data = data.downsample(pre_rebin)
        data = data.dedisperse(dm)
        data = data.downsample(heim_raw_tres*ibox//pre_rebin)
        datats = np.mean(data, axis=0)

        # Low resolution nbeam, nfreq, ntime array
        data_ftb = data.reshape(nfreq_final, data.shape[0]//nfreq_final, data.shape[1]).mean(1)
        # Normalize data excluding outliers
        datatscopy = datats.copy()
        datatscopy.sort()
        medts = np.median(datatscopy[:int(0.975*len(datatscopy))])
        sigts = np.std(datatscopy[:int(0.975*len(datatscopy))])
        datats -= medts 
        datats /= sigts
        beamno_arr.append(beamno)

#        beam_time_arr[beamno, :] = datats
        beam_time_arr[jj, :] = data_ftb 

    return beam_time_arr, multibeam_dm0ts, beamno_arr


def classify_freqtime(fnmodel, dataft):
    """ Function to classify dynspec of candidate. 
    fnmodel can either be a string with the path to
    the keras model or the model itself. 
    """

    if type(fnmodel)==str:
        from keras.models import load_model
        model = load_model(fnmodel)
    else:
        model = fnmodel
        
    mm = np.argmax(dataft.mean(0))
    tlow, thigh = mm-32, mm+32
    if mm<32:
        tlow=0
        thigh=64
    if thigh>dataft.shape[1]:
        thigh=dataft.shape[1]
        tlow=thigh-64
#    dataml = dataft[:,tlow:thigh].copy()
    dataml = dataft
    dataml -= np.median(dataml, axis=1, keepdims=True)
#    dataml /= np.std(dataml, axis=-1)[:, None]
    dataml = dataml/np.std(dataml)
    dataml[dataml!=dataml] = 0.0
    dataml = dataml[None,:,tlow:thigh, None]
    prob = float(model.predict(dataml)[0,1])

    return prob


def filplot(fn, dm, ibox, multibeam=None, figname=None,
             ndm=32, suptitle='', heimsnr=-1,
             ibeam=-1, rficlean=True, nfreq_plot=32, 
             classify=False, heim_raw_tres=1, 
             showplot=True, save_data=True, candname=None,
             fnT2clust=None, imjd=0, fake=False):
    """ Vizualize FRB candidates on DSA-110
    fn is filterbnak file name.
    dm is dispersion measure as float.
    ibox is timecar box width as integer.
    """

    if type(multibeam)==list:
        data_beam_freq_time = []
        nbeam=256
        beam_time_arr_results = generate_beam_time_arr(multibeam, ibox=ibox, pre_rebin=1, dm=dm, heim_raw_tres=heim_raw_tres)
        data_beam_freq_time, _, beamno_arr = beam_time_arr_results
        beam_time_arr = data_beam_freq_time.mean(1)
        multibeam_dm0ts = beam_time_arr.mean(0)
    else:
        beam_time_arr = None
        multibeam_dm0ts = None            
            
    dataft, datadm, tsdm0, dms, datadm0 = proc_cand_fil(fn, dm, ibox, snrheim=-1, 
                                               pre_rebin=1, nfreq_plot=nfreq_plot,
                                               ndm=ndm, rficlean=rficlean,
                                               heim_raw_tres=heim_raw_tres)

    if classify:
        prob = classify_freqtime(MLMODELPATH, dataft)
    else:
        prob = -1
    
    if save_data:
        fnout = (fn.split('/')[-1]).strip('.fil') + '.hdf5'
        fnout = '/home/ubuntu/connor/software/misc/data/MLtraining/' + fnout
        
        paramsdict = {'dm' : dm, 'ibox' : ibox, 'ibeam' : ibeam,
                      'snr' : heimsnr}
        params = np.array([heimsnr, dm, ibox, ibeam, imjd])
        
        g = h5py.File(fnout,'w')
        g.create_dataset('data_freq_time',data=dataft)
        g.create_dataset('data_dm_time',data=datadm)
        if beam_time_arr is None:
            g.create_dataset('data_beam_time',data=[])
        else:
            g.create_dataset('data_beam_time',data=beam_time_arr)
        g.create_dataset('params',data=params)
        g.create_dataset('probability',data=[prob])        
        g.close()
    
    not_real = plotfour(dataft, dataft.mean(0), datadm, datadm0=datadm0, 
                        beam_time_arr=beam_time_arr, figname=figname, dm=dm,
                        dms=[dms[0],dms[-1]], 
                        suptitle=suptitle, heimsnr=heimsnr,
                        ibox=ibox, ibeam=ibeam, prob=prob,
                        showplot=showplot,
                        multibeam_dm0ts=multibeam_dm0ts,fnT2clust=fnT2clust,imjd=imjd,
                        fake=fake)

    return not_real, prob


def filplot_entry(trigger_dict, toslack=True, classify=True,
                  rficlean=False, ndm=32, nfreq_plot=32, save_data=True,
                  fllisting=None):
    """ Given datestring and trigger dictionary, run filterbank plotting, classifying, slack posting.
    Returns figure filename and classification probability. 
    
    Parameters
    ----------
    trigger_dict : dict
        dictionary with candidate parameters, read from json file 
    toslack : bool 
        send plot to slack if real 
    classify : bool 
        classify dynamic spectrum with keras CNN
    ndm : int 
        number of DMs for DM/time plot
    nfreq_plot : int 
        number of freq channels for freq/time plot
    save_data : bool 
        save down classification data
    fllisting : list 
        list of filterbank files 
        
    Returns
    -------
    fnameout : str 
        figure file path
    prob : float 
        probability from dynamic spectrum CNN
    real : bool
        real event, as determined by classfication 
    """

    trigname = list(trigger_dict.keys())[0]
#    trigname = trigger_dict['trigname']  # need to restructure to use this for initial dict
    dm = trigger_dict[trigname]['dm']
    ibox = trigger_dict[trigname]['ibox']
    ibeam = trigger_dict[trigname]['ibeam'] + 1
    timehr = trigger_dict[trigname]['mjds']
    snr = trigger_dict[trigname]['snr']

    if '_inj' in trigname:
        fake = True
        print("This burst was injected")
    else:
        print("Not injected")
        fake = False
    
    fnT2clust = f'{T2dir}/cluster_output.csv'
    fname = None
    if fllisting is None:
        flist = glob.glob(f"{os.path.join(T1dir, trigname)}/*.fil")
        sortlambda = lambda fnfil: int(fnfil.strip('.fil').split('_')[-1])
        fllisting = sorted(flist, key=sortlambda)
    else:
         flist = fllisting

    fname = fllisting[ibeam]

    if toslack:
        showplot = False
    else:
        showplot = True

    # VR hack
    try:
        ra_mjd, dec_mjd = dsautils.coordinates.get_pointing(ibeam, obstime=Time(timehr, format='mjd'))
        l, b = dsautils.coordinates.get_galcoord(ra_mjd.value, dec_mjd.value)
    except:
        ra_mjd = 1.0*u.deg
        dec_mjd = 71.5*u.deg
        l = 100.0
        b = 50.0

#    ind_near = utils.match_pulsar(ra_mjd, dec_mjd, thresh_deg=3.5)

#    psr_txt_str = ''
#    for ind_jj in ind_near:
#        name_psrb = utils.query['PSRB'][ind_jj]
#        name_psrj = utils.query['PSRJ'][ind_jj]
#        dm_psr = utils.query['DM'][ind_jj]
#        psr_txt_str += '%s/%s: %0.1f\n' % (name_psrb, name_psrj, dm_psr)

    outstr = (trigname, dm, int(ibox), int(ibeam), timehr, ra_mjd.value, dec_mjd.value, l, b)
    suptitle = 'candname:%s  DM:%0.1f  boxcar:%d \nibeam:%d MJD:%f \nRa/Dec=%0.1f,%0.1f Gal lon/lat=%0.1f,%0.1f' % outstr

    figdirout = webPLOTDIR
    figname = figdirout+trigname+'.png'

    assert fname is not None, "Must set fname"
    not_real, prob = filplot(fname, dm, ibox, figname=figname,
                             ndm=ndm, suptitle=suptitle, heimsnr=snr,
                             ibeam=ibeam, rficlean=rficlean, 
                             nfreq_plot=nfreq_plot, classify=classify, showplot=showplot, 
                             multibeam=flist, heim_raw_tres=1, save_data=save_data,
                             candname=trigname, fnT2clust=fnT2clust, imjd=timehr,
                             fake=fake)
    real = not not_real

    if toslack:
        if not_real==False:
            print("Sending to slack")
            slack_client.files_upload(channels='candidates', file=figname, initial_comment=figname)
        else:
            print("Not real. Not sending to slack")

    return figname, prob, real
