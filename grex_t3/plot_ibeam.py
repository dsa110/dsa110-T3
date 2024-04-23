import sys

import glob
import numpy as np
import matplotlib.pylab as plt

import filplot_funcs

if __name__=='__main__':

    if len(sys.argv)!=6:
        print("Expected: datestr candname ibeam ibox dm")
        exit()
    BASEDIR='/data/dsa110/T1/'
    datestr = sys.argv[1]
    candname = sys.argv[2]
    ibeam = int(sys.argv[3])
    ibox = int(sys.argv[4])
    dm = float(sys.argv[5])
    fpath = BASEDIR+'corr*/'+datestr+'/'+'fil_'+candname+'/*_'+str(ibeam)+'*.fil'
    fl = glob.glob(fpath)    
    print(fl, fpath)
    fn = fl[0]
    print(fn)
    fnout = fn.split('/')[-1]
    dataft, datadm, tsdm0, dms, datadm0 = filplot_funcs.proc_cand_fil(fn, dm, 1, snrheim=-1,
                                               pre_rebin=1, nfreq_plot=1024,
                                                                      ndm=256, rficlean=False, norm=True, freq_ref=1400.)
#    mm = np.std(dataft,1)
#    dataft[mm>1.265] = 0.0

    fnout = '/home/ubuntu/connor/data/%s_rficlean.npy' %  fn.split('/')[-1]
    print('Saving high res data to:\n%s' % fnout)
    np.save(fnout, dataft)
    dataft = dataft.downsample(ibox)
    dataft = dataft.reshape(dataft.shape[0]//32, 32, dataft.shape[-1]).mean(1)
    
    nfreq, ntime = dataft.shape
    xminplot,xmaxplot = 400,600 # milliseconds
    xminplot,xmaxplot = 500.-300*ibox/16.,500.+300*ibox/16 # milliseconds                                             
    if xminplot<0:
        xmaxplot=xminplot+500+300*ibox/16
        xminplot=0
    tmin, tmax = 0., 1e3*dataft.header['tsamp']*ntime
    tarr = np.linspace(tmin, tmax, ntime)
    freqmax = dataft.header['fch1']
    freqmin = freqmax + dataft.header['nchans']*dataft.header['foff']

    dm_min, dm_max = dms.min(), dms.max()
    extentft=[tmin,tmax,freqmin,freqmax]
    extentdm=[tmin, tmax, dm_min, dm_max]
    
    fig = plt.figure(figsize=(8,10))
    plt.subplot(311)
    plt.imshow(dataft-np.mean(dataft,axis=1,keepdims=True), aspect='auto', extent=extentft)
    plt.xlim(xminplot,xmaxplot)
    plt.xlabel('Time (ms)')
    plt.ylabel('Freq ')
    plt.subplot(312)
    plt.plot(tarr, dataft.mean(0))
    plt.xlim(xminplot,xmaxplot)    
    plt.xlabel('Time (ms)')
    plt.subplot(313)
    plt.imshow(datadm, aspect='auto', extent=extentdm)
    plt.xlim(xminplot,xmaxplot)
    plt.suptitle('%s candname:%s \nDM:%0.1f boxcar:%d ibeam:%d' % (datestr, candname, dm, ibox, ibeam), color='C1')
    plt.show()

    
