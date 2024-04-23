import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
from astropy import units as u
import glob
import time

import candproc_tools

def plotfour_grex(cand):
    """ Plot the four plots for a candidate.
    """
    cand_dir = '/hdd/data/candidates/T3/candplots/'
    fignameout = cand_dir + cand.your_file.split('/')[-1].strip('.fil') + '.pdf'

    fncluster = '/hdd/data/candidates/T2/cluster_output.csv'
    cluster_output = pd.read_csv(fncluster)
    ind_clust = np.where(np.abs(cluster_output['mjds'] - 10.) < 10.0)[0]
    cluster_output = cluster_output.loc[ind_clust]

    ntime, nchans = cand.data.shape[0], cand.data.shape[1]
    times = np.linspace(0,cand.tsamp*ntime,ntime)
    freqs = np.linspace(cand.fch1, cand.fch1+cand.foff*nchans, nchans)

    data_timesteam = cand.dedispersed.mean(1)
    mm = np.argmax(data_timesteam)
    data_timesteam = data_timesteam[mm-256:mm+256]
    data_freqtime = cand.dedispersed[mm-256:mm+256, 300:-300]
    data_freqtime = data_freqtime - np.mean(data_freqtime, 
                                            axis=0, keepdims=True)
    data_freqtime = data_freqtime.T
    data_dmt = cand.dmt[:, mm-256:mm+256]
    data_dmt = data_dmt - np.mean(data_dmt, axis=1, keepdims=True)

    times = times[mm-256:mm+256]
    tmin, tmax = times[0], times[-1]
    freqmin, freqmax = freqs[0], freqs[-1]

    fig = plt.figure(figsize=(8,10))
    plt.subplot(221)
    extentft=[tmin,tmax,freqmin,freqmax]
    plt.imshow(data_freqtime, aspect='auto', 
               interpolation='nearest', extent=extentft)
    plt.xlabel('Time (s)')
    plt.ylabel('Freq (MHz)')

    plt.subplot(222)
    extentdmt=[tmin, tmax, 0, 1]
    plt.imshow(data_dmt, aspect='auto', 
               interpolation='nearest', extent=extentdmt)
    plt.xlabel('Time (s)')
    plt.ylabel('DM (pc cm^-3)')

    plt.subplot(223)
    plt.plot(times, data_timesteam)
    plt.xlabel('Time (s)')

    plt.subplot(224)
    plt.scatter(cluster_output['mjds'], cluster_output['dm'],
                s=1,c=cluster_output['snr'], vmin=8, vmax=50.)
    plt.grid('on', alpha=0.25)
    plt.ylabel('DM (pc cm^-3)')
    plt.xlabel('Time (MJD)')

    plt.savefig(fignameout)


def proc_and_plot_candidate(fnvoltage):
    fnfil = 'test.fil'

    if not os.path.exists(fnvoltage):
        print("ERR: File %s does not exist" % fnvoltage)
        return

    stokesi_obj = candproc_tools.read_voltage_data(fnvoltage, 
                                        timedownsample=2**4,
                                        freqdownsample=None)
    print('Finished reading in %s' % fnvoltage)
    
    candproc_tools.write_sigproc(fnfil, stokesi_obj)
    print('Wrote to %s' % fnfil)

    cand = candproc_tools.read_proc_fil(fnfil, dm=100.0, tcand=10.4,width=1,
                             device=0, tstart=0,tstop=20,
                             ndm=32, dmtime_transform=True)
    print('Finished dedispersion')
    plotfour_grex(cand)

def plotfour(dataft, datadmt, cand_heim,
             beam_time_arr=None, figname_out=None, dm=0,
             dms=[0,1], 
             datadm0=None, suptitle='', heimsnr=-1,
             ibox=1, ibeam=-1, prob=-1,
             showplot=True, multibeam_dm0ts=None,
             time_sec_ii=0.0):
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
        figname_out : 
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
    datats_HR = dataft.mean(0)
    datats_HR /= np.std(datats_HR[datats_HR!=np.max(datats_HR)])
    nfreq, ntime = dataft.shape
    dataft_HR = dataft.copy()
    dataft = dataft[:, :ntime//ibox*ibox].reshape(nfreq, ntime//ibox, ibox).mean(-1)
    ntime = dataft.shape[1]
    dataft.header['tsamp'] = dataft.header['tsamp']*ibox
    datats = dataft.mean(0)
    datats /= np.std(datats[datats!=np.max(datats)])

    xminplot,xmaxplot = 500.-300*ibox/16.,500.+300*ibox/16 # milliseconds
    if xminplot<0:
        xmaxplot=xminplot+500+300*ibox/16        
        xminplot=0
    dm_min, dm_max = dms[0], dms[1]
    tmin, tmax = 0., 1e3*dataft.header['tsamp']*ntime
    xminplot,xmaxplot = 0, tmax

    freqmax = dataft.header['fch1']
    freqmin = freqmax + dataft.header['nchans']*dataft.header['foff']
    freqs = np.linspace(freqmin, freqmax, nfreq)
    tarr = np.linspace(tmin, tmax, ntime)
    tarr_HR = np.linspace(tmin, tmax, dataft_HR.shape[1])
    fig = plt.figure(figsize=(8,10))

    plt.subplot(221)
    extentft=[tmin,tmax,freqmin,freqmax]
    plt.imshow(dataft, 
               aspect='auto',
               extent=extentft, 
               interpolation='nearest')
    DM0_delays = xminplot + dm * 4.15E6 * (freqmin**-2 - freqs**-2)
    plt.plot(DM0_delays, freqs, c='r', lw='2', alpha=0.35)
    plt.xlim(xminplot,xmaxplot)
    plt.xlabel('Time (ms)')
    plt.ylabel('Freq (MHz)')
    if prob!=-1:
        plt.text(xminplot+50*ibox/16.,0.5*(freqmax+freqmin),
                 "Prob=%0.2f" % prob, color='white', fontweight='bold')
        classification_dict['prob'] = prob

    plt.subplot(222)
    extentdm=[tmin, tmax, dm_min, dm_max]
    plt.imshow(datadmt[::-1], aspect='auto',extent=extentdm)
    plt.xlim(xminplot,xmaxplot)
    plt.xlabel('Time (ms)')
    plt.ylabel(r'DM (pc cm$^{-3}$)')

    plt.subplot(223)
    plt.plot(tarr_HR, datats_HR)
    plt.plot(tarr, datats)

    plt.grid('on', alpha=0.25)
    plt.xlabel('Time (ms)')
    plt.ylabel(r'Power ($\sigma$)')
    plt.xlim(xminplot,xmaxplot)
    plt.text(0.51*(xminplot+xmaxplot), 
            0.5*(max(datats)+np.median(datats)), 
            'Heimdall S/N : %0.1f\nHeimdall DM : %d\
            \nHeimdall ibox : %d\nibeam : %d' % (heimsnr,dm,ibox,ibeam), 
            fontsize=8, verticalalignment='center')
    
        
#        plt.subplot(326)
#        plt.plot(np.linspace(freqmax,freqmin,datadm0.shape[0]), np.mean(datadm0,axis=-1), color='k')

#        plt.semilogy()
#        plt.legend(['spectrum'], loc=2)
#        plt.xlabel('freq [MHz]')

    plt.subplot(224)
    # plt.scatter(cand_heim['time_sec'], cand_heim['dm'], 3, 
    #             c=cand_heim['snr'], 
    #             cmap='RdBu', vmax=15., vmin=6.0)
    # plt.colorbar(label=r'SNR')
    # plt.scatter(time_sec_ii, dm,
    #             s=100, marker='s',
    #             facecolor='none', edgecolor='black')
    # plt.xlabel('Time (s)')
    # plt.ylabel('DM')

    # plt.tight_layout()
    if figname_out is not None:
        try:
            plt.savefig(figname_out)
        except:
            print("\n ERR: COULD NOT SAVE FIGURE")
    if showplot:
        try:
            plt.show()
        except:
            print("\n ERR: COULD NOT SHOW FIG")

    # return not_real
        
def dm_transform(data, dm_max=20,
                 dm_min=0, dm0=None, ndm=64, 
                 freq_ref=None, downsample=16):
    """ Transform freq/time data to dm/time data.                                                                                                                                           
    """
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

def proc_candidate(fncand='/home/user/cand_times_sync/heimdall.cand', 
                   mkplot=True, ndm=32, dmtrans=True, target_DM=None, 
                   target_RA=None, showplot=False):
    if type(fncand)==str:
        cand_heim = staretools.read_heim_pandas(fncand, skiprows=0)
        mjd = staretools.get_mjd_cand_pd(cand_heim)
    elif type(fncand)==staretools.pd.core.frame.DataFrame:
        cand_heim = fncand 
        mjd = staretools.get_mjd_cand_pd(cand_heim)

#    ind = np.where(mjd>(Time.now().mjd-3))[0]
    station = cand_heim['station']
    norm=True

    for ii in range(len(cand_heim)):
        if target_RA is not None:
            ra_diff = staretools.get_RA_diff(mjd.iloc[ii], staretools.sgr_1935_ra)

        station_ii = station.iloc[ii]
        dm_ii = cand_heim['dm'].iloc[ii]
        cand_name_ii = cand_heim['cand'].iloc[ii]
        mjd_ii = mjd.iloc[ii]
        ibox = int(2**(cand_heim['log2width'].iloc[ii]))

        if target_RA is not None and abs(ra_diff.deg)>90.:
#            print("Skipping, SGR1935 not in beam")
            continue

        if target_DM is not None:
            if np.abs(dm_ii-staretools.dm_sgr1935)>20.0:
#                print("Skipping, DM is not close to 332 pc cm**-3")
                continue

        if type(fncand)==str:
            fig_dir = fncand.split('coin')[0]+'/plots/'
        else:
            fig_dir = './'

        if not os.path.isdir(fig_dir):
            os.system('mkdir %s' % fig_dir)

        figname_out = fig_dir + '%0.6f-station:%d--dm%0.1f.png' % (mjd_ii, station_ii, dm_ii)

        if os.path.exists(figname_out):
            print("Skipping candidates %s, figure already exists" % cand_name_ii)
            continue

        if station_ii==1:
            try:
                fnfil = glob.glob('/home/user/candidates/candidate_%d.fil'%cand_name_ii)[0]
            except:
                print("Could not find candidate_%d.fil"%cand_name_ii)
                continue        
        elif station_ii==2:
            cand_dir_2 = '/home/user/cand_fil_station_2/'
            fnfil = cand_dir_2 + 'candidate_%d.fil' % cand_name_ii
            if not os.path.exists(fnfil):
                try:
                    os.system('scp user@166.140.120.248:/home/user/candidates/candidate_%d.fil %s'%(cand_name_ii,cand_dir_2))
                except:
                    print("Could not scp candidate_%d.fil" % cand_name_ii)
                    continue
        elif station_ii==3:
            cand_dir_3 = '/home/user/cand_fil_station_3/'
            fnfil = cand_dir_3 + 'candidate_%d.fil' % cand_name_ii
            if not os.path.exists(fnfil):
                try:
                    os.system('scp user@158.154.14.10:/home/user/candidates/candidate_%d.fil %s'%(cand_name_ii,cand_dir_3))
                except:
                    print("Could not scp candidate_%d.fil" % cand_name_ii)
                    continue
        
        if not os.path.exists(fnfil):
            continue

        print("\nReading in %s"%fnfil)
        data, freq, dt, header = staretools.read_fil_data_stare(fnfil, 0, -1)
        nf, nt = data.shape 

        pstr = (station_ii, cand_name_ii, dm_ii, ibox, mjd_ii)
        print("     Station:%d cand:%d DM:%0.2f ibox:%d mjd:%0.7f" % pstr)
        if norm:
            data = data-np.median(data,axis=1,keepdims=True)
            data /= np.std(data)

        if dmtrans:
            print("     Applying DM/Time transform")
            dm_err = ibox / 1.0 * 25.
            dm_err = 25.
            datadm, dms = dm_transform(data, dm_max=dm_ii+dm_err,
                                       dm_min=dm_ii-dm_err, dm0=dm_ii, 
                                       ndm=ndm, 
                                       freq_ref=np.mean(freq),
                                       downsample=max(4,ibox))
        else:
            datadm = None

        data = data.dedisperse(dm_ii)
        data = data - np.mean(data, axis=-1, keepdims=True)
        data = data.reshape(nf//16, 16, -1).mean(1)

        ind_tt = np.where(abs(cand_heim['time_sec']-cand_heim['time_sec'][ii])<10)
        cand_heim_ii = cand_heim.loc[ind_tt]

        if mkplot is True:
            plotfour(data, datadm, cand_heim_ii,
                     figname_out=figname_out, dm=dm_ii,
                     dms=dms,
                     suptitle='', heimsnr=-1, showplot=showplot,
                     ibox=ibox, ibeam=-1, prob=-1,
                     multibeam_dm0ts=None,
                     time_sec_ii=cand_heim['time_sec'][ii])

if __name__=='__main__':
    fncand = sys.argv[1] + '/coincidence_3stations.csv'
    proc_candidate(fncand=fncand,
                   mkplot=True, ndm=32, dmtrans=True, target_DM=None, 
                   target_RA=None, showplot=False)

    if False:
        nday_lookback = 20
        rsync_heimdall_cand()
        fncand1 = '/home/user/cand_times_sync/heimdall.cand'
        fncand2 = '/home/user/cand_times_sync_od/heimdall_2.cand'
        fncand3 = '/home/user/cand_times_sync/heimdall_3.cand'

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

        ind1 = np.where(mjd_arr_1>(Time.now().mjd-nday_lookback))[0]
        ind2 = np.where(mjd_arr_2>(Time.now().mjd-nday_lookback))[0]
        ind3 = np.where(mjd_arr_3>(Time.now().mjd-nday_lookback))[0]

        data_1, mjd_arr_1 = data_1.loc[ind1], mjd_arr_1[ind1]
        data_2, mjd_arr_2 = data_2.loc[ind2], mjd_arr_2[ind2]
        data_3, mjd_arr_3 = data_3.loc[ind3], mjd_arr_3[ind3]

        data_1.insert(0, 'station', 1)
        data_2.insert(0, 'station', 2)
        data_3.insert(0, 'station', 3)

        proc_candidate(fncand=data_1,
                       mkplot=True, ndm=32, dmtrans=True, 
                       target_RA=staretools.sgr_1935_ra, 
                       target_DM=staretools.dm_sgr1935, 
                       showplot=False)

        proc_candidate(fncand=data_2,
                       mkplot=True, ndm=32, dmtrans=True, 
                       target_RA=staretools.sgr_1935_ra, 
                       target_DM=staretools.dm_sgr1935, 
                       showplot=False)

        proc_candidate(fncand=data_3,
                       mkplot=True, ndm=32, dmtrans=True, 
                       target_RA=staretools.sgr_1935_ra, 
                       target_DM=staretools.dm_sgr1935, 
                       showplot=False)

        first_MJD = min(mjd_arr_1)
        last_MJD = max(mjd_arr_1)








