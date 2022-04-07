import sys

import glob
import numpy as np
import matplotlib.pylab as plt

import filplot_funcs

if __name__=='__main__':
    BASEDIR='/data/dsa110/T1/'
    datestr = sys.argv[1]
    candname = sys.argv[2]
    ibeam = int(sys.argv[3])
    ibox = int(sys.argv[4])
    dm = float(sys.argv[5])
    fpath = BASEDIR+'corr*/'+datestr+'/'+'fil_'+candname+'/*_'+str(ibeam)+'*.fil'
    fl = glob.glob(fpath)    
#    fn, dm, ibox = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
    fn = fl[0]
    fnout = fn.split('/')[-1]
    fnfigout = '/home/ubuntu/connor/data/%s.png' %  fn.split('/')[-1]
    dataft, datadm, tsdm0, dms, datadm0 = filplot_funcs.proc_cand_fil(fn, dm, ibox, snrheim=-1,
                                               pre_rebin=1, nfreq_plot=64,
                                               ndm=64, rficlean=False, norm=True)
    dataftx, datadmx, tsdm0x, dmsx, datadm0x = filplot_funcs.proc_cand_fil(fn, dm, ibox, snrheim=-1,
                                               pre_rebin=1, nfreq_plot=64,
                                               ndm=64, rficlean=True, norm=True)


    nfreq, ntime = dataft.shape
    xminplot,xmaxplot = 200,800 # milliseconds                                                                            
    tmin, tmax = 0., 1e3*dataft.header['tsamp']*ntime
    tarr = np.linspace(tmin, tmax, ntime)
    freqmax = dataft.header['fch1']
    freqmin = freqmax + dataft.header['nchans']*dataft.header['foff']

    dm_min, dm_max = dms.min(), dms.max()
    extentft=[tmin,tmax,freqmin,freqmax]
    extentdm=[tmin, tmax, dm_min, dm_max]
    
    fig = plt.figure(figsize=(11,6))
    plt.subplot(121)
    plt.title('Not RFI Cleaned')
    plt.imshow(dataft, aspect='auto', extent=extentft, vmax=3, vmin=-1)
    plt.xlabel('Time (ms)')
    plt.ylabel('Freq ')
    plt.subplot(122)
    plt.title('RFI Cleaned')
    plt.imshow(dataftx, aspect='auto', extent=extentft, vmax=3, vmin=-1)    
    plt.xlabel('Time (ms)')
    plt.xlim(0,tarr.max())
    plt.suptitle(fn, color='C1')
    plt.savefig(fnfigout)
    #plt.show()
    
