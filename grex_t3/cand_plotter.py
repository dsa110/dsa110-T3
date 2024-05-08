### Plotting T3 candidates 04/18/24 updated
### time gap between cand MJD and voltage starting time still exists
### usage: poetry run python cand_plotter.py <full .json name>


import numpy as np
import pandas as pd
import sys
from matplotlib import pyplot as plt
import xarray as xr
import json
from astropy.time import Time
import os
import logging
from your.formats.filwriter import make_sigproc_object

T3_path = '/home/user/zghuai/GReX-T3/grex_t3/'
sys.path.append(T3_path)
import candproc_tools as ct
import analysis_tools as at


logfile = '/home/user/zghuai/GReX-T3/services/T3_plotter.log'
logging.basicConfig(filename=logfile,
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logging.info('Starting cand_plotter.py!')

### To run cand_plotter.py independently, uncomment this line.
js = str(sys.argv[1]) # full .json filename


def get_cand(JSON):
    """
    Reads the input json file
    Returns a table containing ('mjds' 'snr' 'ibox' 'dm' 'ibeam' 'cntb' 'cntc' 'specnum')
    """
    f = open('/hdd/data/candidates/T2/'+JSON)
    data = json.load(f)
    tab = pd.json_normalize(data[JSON.split(".")[0]], meta=['id'])
    
    f.close()
    return(tab)
    

def gen_cand(fn_vol, fn_tempfil, fn_filout, JSON): # tab - json file
    """
    Reads in a raw voltage file
    Calculates Stokes I 
    Picks a window given ToA and saves to a temporary .fil
    Dedisperses the window, Generates DMtime
    Saves dedispersed window to a .fil file
    Downsamples the window
    ----------
    Inputs:
    fn_vol = input voltage filename
    fn_tempfil = temporary filterbank file, will be removed after generating a plot
    fn_filout = output .fil filename
    JSON = candidate .json filename (e.g. 240321aazm.json)
    ----------
    Returns:
    cand = dedispersed, downsampled Candidate object by the YOUR package
    T0 = start time of the input voltage file in MJD
    tab = .json table
    """

    tab = get_cand(JSON)
    # t0 = tab["mjds"].values[0] # ToA of the candidate
    dt = 8.192e-6 # s 
    Dt = 4.15 * tab["dm"].values[0] * (1/1.28**2 - 1/1.53**2) / 1e3 # delay in seconds given DM
    
    (stokesi, T0, dur) = ct.read_voltage_data(fn_vol, timedownsample=None, freqdownsample=None, verbose=True, nbit='uint16')
    print('Done reading .nc and calc stokes I')

    window_width = int(Dt*2 / dt) # number of samples in the pulse window
    print('window size = ', window_width)
    # mm = int(stokesi.shape[0]/2) # temporarily choose the center of stokesi  # (t0-T0)*86400/dt 
    t_in_nc = (tab['mjds'].values[0]-T0)*86400 
    print('time = ', t_in_nc)
    print('Duration = ', dur)
    dur = int(dur/dt)
    print('samples = ', dur)
    print('stokesi shape = ', stokesi.shape)
    if t_in_nc > dur*dt: # not captured in .nc, pick a random time interval to plot for now, need to log errors later.
        mm = stokesi.shape[0]//2
        print('not in .nc; mm = ', mm)
    else: # if in the .nc file
        mm = int(t_in_nc / dt) # number of samples after the start of .nc file
        print('in .nc! mm = ', mm)
        if (mm<(window_width//2)):
            window_width = int(2*mm)
            print('toward the beginning')
        elif ((dur-mm)<window_width//2):
            window_width = (dur-mm)
            print('toward the end', dur-mm)

    if window_width < 10:
        print("Too few samples!")
        return
    print('mm = ', mm, 'windowsize = ', window_width)
    logging.info(f"in gen_cand, mm = {mm}")
    
    cand_disp = stokesi[(mm-window_width//2):(mm+window_width//2), :] # dispersed candidate in xarray dataarray format.

    print('start time = ', T0+(mm-window_width//2)*dt/86400)
    print(cand_disp.shape)
    ct.write_sigproc(fn_tempfil, cand_disp, t_start=T0+(mm-window_width//2)*dt/86400) 
    print('Done writing to a temporary .fil')

    
    # Using Liam's candproc_tools.py to read the temporary .fil, dedisperse, and calculate DMtime
    cand = ct.read_proc_fil(fnfil=fn_tempfil, 
                            dm=tab["dm"].values[0], 
                            tcand=2.0, 
                            width=1, 
                            device=0, 
                            tstart=0,
                            tstop=(cand_disp.time.max()-cand_disp.time.min()).values*86400,
                            zero_topbottom=False,
                            ndm=32, 
                            dmtime_transform=True)
    print('Done dedispersing')

    # generate a smaller window containing the dedispersed pulse and save to .fil
    # (before downsampling)
    window_time = 128 * tab['ibox'].values[0] 
    data_timestream = cand.dedispersed.mean(1)
    mm = np.argmax(data_timestream) # eventually this will be replaced with actual ToA 
    data_freqtime = cand.dedispersed[mm-window_time//2:mm+window_time//2, :] 

    # write to .fil
    nchans = cand.nchans
    foff = cand.foff
    fch1 = cand.fch1
    tsamp = cand.tsamp 
    sigproc_object = make_sigproc_object(
                                    rawdatafile=fn_filout,
                                    source_name="bar",
                                    nchans=nchans,
                                    foff=foff,  # MHz
                                    fch1=fch1,  # MHz
                                    tsamp=tsamp,  # seconds
                                    tstart=59246,  # MJD need to update this!!!
                                    src_raj=112233.44,  # HHMMSS.SS
                                    src_dej=112233.44,  # DDMMSS.SS
                                    machine_id=0,
                                    nbeams=1,
                                    ibeam=0,
                                    nbits=16,
                                    nifs=1,
                                    barycentric=0,
                                    pulsarcentric=0,
                                    telescope_id=6,
                                    data_type=0,
                                    az_start=-1,
                                    za_start=-1,)
    
    sigproc_object.write_header(fn_filout)
    sigproc_object.append_spectra(data_freqtime, fn_filout)

    # downsampling 
    cand.decimate(key = 'ft', 
                  decimate_factor = 16, 
                  axis = 1) # frequency ds by 16
    cand.decimate(key = 'ft', 
                  decimate_factor = tab['ibox'].values[0],
                  axis = 0,
                  pad = True) # time ds
    cand.decimate(key='dmt',
                  decimate_factor = tab['ibox'].values[0],
                  axis = 1,
                  pad = True) # time ds in DMtime domain
    cand.tsamp = cand.tsamp * tab['ibox'].values[0] # update downsampled time resolution 
    print('Done downsampling')
    print(cand.dedispersed.shape)
    print(cand.dmt.shape)
    return(cand, T0, dur*dt, tab)



def plot_grex(cand, T0, dur, tab, JSON): 

    """
    Plots:
        downsampled, dedispersed pulse, 
        DM vs time, 
        nearby candidates from cluster_output.csv within a larger time window.
    ----------
    Inputs:
    cand = downsampled, dedispersed candidate object, the first output from gen_cand() function
    T0 = start time of the voltage file in MJD from gen_cand()
    dur = duration of the entire netcdf file in seconds
    tab = candidate .json table from gen_cand()
    JSON = .json filename
    ----------
    Returns None
    """
    
    window_time = 1280 # number of samples in the downsampled window
    ntime, nchans = cand.dedispersed.shape[0], cand.dedispersed.shape[1]
    f_low = int(278/16) # roughly from 1300MHz to 1500MHz, removing junks near the two edges.
    f_high = nchans - int(164/16)    
    
    cluster = pd.read_csv("/hdd/data/candidates/T2/cluster_output.csv")
    # snr,if,specnum,mjds,ibox,idm,dm,ibeam,cl,cntc,cntb,trigger
    this_cand = np.where(np.abs(cluster['mjds']-tab["mjds"].values[0])<30./86400)[0] # candidates nearby within 30s
    
    data_timestream = cand.dedispersed.mean(1)


    t_in_nc = (tab['mjds'].values[0] - cand.tstart) * 86400 # seconds after the start of the intermediate .fil
    print('Time in .nc = ', t_in_nc)
    l = cand.dedispersed.shape[0] # length of .fil file
    print('length of timestream = ', l)
    if t_in_nc > l*cand.tsamp: # not captured in .fil, pick a random time interval
        mm = len(data_timestream)//2 # update this and log errors. 
        print('not in .nc; mm = ', mm)
    else: # if in the .nc file
        mm = int(t_in_nc / cand.tsamp)
        print('in .nc! mm = ', mm)
        if (mm<(window_time//2)):
            window_time = int(2*mm)
            print('toward the beginning')
        elif ((l-mm)<window_time//2):
            window_time = (l-mm)
            print('toward the end')

    print('mm, window = ', mm, window_time)

    if window_time<4:
        print('Too few samples to plot')
        return

    logging.info(f"in plotting, mm = {mm}, time window = {window_time}")

    data_timestream = data_timestream[mm-window_time//2:mm+window_time//2]
    # Dedispersed pulse 
    data_freqtime = cand.dedispersed[mm-window_time//2:mm+window_time//2, f_low:f_high] # roughly from 1300MHz to 1500MHz
    data_freqtime = (data_freqtime - 
                     np.mean(data_freqtime, axis=0, keepdims=True))
    data_freqtime = data_freqtime.T

    # DM time
    data_dmt = cand.dmt[:, mm-window_time//2:mm+window_time//2]
    data_dmt = (data_dmt - 
                np.mean(data_dmt, axis=1, keepdims=True))

    # Construct time array for the window
    times = np.linspace(0,cand.tsamp*ntime,ntime) * 1e3 # Convert into milliseconds
    times = times[mm-window_time//2:mm+window_time//2]
    tmin, tmax = times[0], times[-1]
    # Construct the downsampled frequency array, after truncating highest and lowest edges
    freqs = np.linspace(cand.fch1+(nchans-f_low)*cand.foff*16, cand.fch1+cand.foff*16*(nchans-f_high), f_high-f_low)
    freqmin, freqmax = freqs[0], freqs[-1]

    # Calculate std of the given window.
    snr_tools = at.SNR_Tools()
    stds = snr_tools.calc_snr_presto(data_timestream, verbose=True)[1]
    print(stds)
    
    # Plot
    logging.info("Starting to plot!")
    fig = plt.figure(figsize=(10,15))
    grid = plt.GridSpec(9, 6)

    # first row, collapse frequency 
    plt.subplot(grid[0, :6])
    plt.plot(times, (data_timestream-np.mean(data_timestream))/stds, lw=1., color='black')
    plt.ylabel('SNR')
    plt.xlim(times.min(), times.max())
    plt.xticks([], [])

    # the dedispersed pulse
    plt.subplot(grid[1:3, :6])
    vmax = np.mean(data_freqtime) + 2*np.std(data_freqtime)
    plt.imshow(data_freqtime, 
               aspect='auto', 
               vmax=vmax,
               extent=(times.min(), times.max(), freqs.min(), freqs.max()),
               interpolation='nearest')
    DM0_delays = tmin + cand.dm * 4.15E6 * (freqmin**-2 - freqs**-2) # zero DM sweep
    plt.plot(DM0_delays, freqs, c='r', lw='2', alpha=0.35)
    plt.xlabel('Time (ms)+ MJD {}'.format(cand.tstart), fontsize=12)
    plt.ylabel('Frequency (MHz)')
    plt.xlim(tmin, tmax)

    # DM vs. time
    plt.subplot(grid[4:6, 0:6])
    plt.imshow(data_dmt, 
               aspect='auto', 
               interpolation='nearest',
               extent=(times.min(), times.max(), 0, 2*cand.dm))
    plt.xlabel('Time (ms)+ MJD {}'.format(cand.tstart), fontsize=12)
    plt.ylabel('DM')
    
    # DM vs. MJD in cluster_output.csv
    plt.subplot(grid[7:9, 0:6])
    plt.scatter(cluster["mjds"][this_cand].values, 
                cluster["dm"][this_cand].values, 
                c=cluster['snr'][this_cand].values)
    plt.xlabel('Time (MJD)', fontsize=12)
    plt.ylabel('DM')

    # doesn't seem to work?
    plt.subplots_adjust(left=0.1,
                        bottom=0.1, 
                        right=0.8, 
                        top=0.8, 
                        wspace=0.05, 
                        hspace=0.01)

    # Add some candidate description as the tile
    fig.text(0.1, 0.875, 'DM = {} pc/cm^3'.format(tab["dm"].values[0]),
             fontsize = 12,fontweight='semibold')
    fig.text(0.1, 0.86, 'Arriving time = MJD {}'.format(tab['mjds'].values[0]),
             fontsize = 12,fontweight='semibold')
    fig.text(0.1, 0.845, 'SNR = {}'.format(tab['snr'].values[0]),
             fontsize = 12,fontweight='semibold')
    fig.text(0.1,0.83, 'Filename:'+JSON,
             fontsize = 12,fontweight='semibold')
    
    # if ((tab['mjds'].values[0]-T0)*86400>dur):
    #     fig.text(0.55, 0.875, "Candidate not in this NetCDF file!", color='red')

    plt.savefig('/hdd/data/candidates/T3/candplots/grex_cand{}.png'.format(JSON.split('.')[0]), bbox_inches='tight')
    
    plt.show()

    return()

### To run cand_plotter.py independently.
if __name__ == '__main__':
    c = js.split('.')[0] # candidate name 
    v = "/hdd/data/voltages/grex_dump-"+c+".nc" # voltage file
    fn_tempfil = "/hdd/data/candidates/T3/candplots/intermediate.fil" # output temporary .fil
    fn_outfil = f"/hdd/data/candidates/T3/cand{c}.fil" # output dedispersed candidate .fil
    (cand, T0, dur, tab) = gen_cand(v, fn_tempfil, fn_outfil, c+'.json')

    plot_grex(cand, T0, dur, tab, c+".json") 

    # cmd = "rm {}".format(fn_tempfil)
    # print(cmd)
    # os.system(cmd)

