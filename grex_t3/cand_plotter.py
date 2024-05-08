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
    try:
        f = open('/hdd/data/candidates/T2/'+JSON)
        data = json.load(f)
        tab = pd.json_normalize(data[JSON.split(".")[0]], meta=['id'])
        f.close()
        return tab
        
    except Exception as e:
        print(f"Error loading JSON file: {e}") # log error message
        logging.error("Error getting candidate json file: %s", str(e))
        return None  
    

def gen_cand(fn_vol, fn_tempfil, fn_filout, JSON, v=False): # tab - json file
    """
    ----------
    Inputs:
    fn_vol = input voltage filename
    fn_tempfil = temporary filterbank filename, will be removed after generating a plot
    fn_filout = output .fil filename
    JSON = candidate .json filename (e.g. 240321aazm.json)
    v = True: log verbose information
    ----------
    Returns:
    cand = dedispersed, downsampled Candidate object usinng the YOUR package
    tab = .json table
    """

    tab = get_cand(JSON)
    # t0 = tab["mjds"].values[0] # ToA of the candidate
    dt = 8.192e-6 # s 
    Dt = 4.15 * tab["dm"].values[0] * (1/1.28**2 - 1/1.53**2) / 1e3 # delay in seconds given DM
    
    (stokesi, T0, dur) = ct.read_voltage_data(fn_vol, 
                                              timedownsample=None, 
                                              freqdownsample=None, 
                                              verbose=True, 
                                              nbit='uint16')
    print('Done reading .nc and calc stokes I')

    # number of samples in the pulse window
    window_width = int(Dt*2 / dt) 
    print('window size = ', window_width)
    # ToA of the pulse in the netcdf file (in seconds)
    t_in_nc = (tab['mjds'].values[0]-T0)*86400 
    print('time = ', t_in_nc)
    print('Duration = ', dur)
    # duration of the entire netcdf file in number of time samples
    dur = int(dur/dt)
    print('samples = ', dur)
    print('stokesi shape = ', stokesi.shape)

    if t_in_nc > dur*dt: # not captured in .nc, pick a random time interval to plot for now, need to log errors later.
        # mm = stokesi.shape[0]//2
        logging.shutdown("Cand MJD not in the NetCDF file, shutting down.")
        return 
        # print('not in .nc; mm = ', mm)
    # if in the .nc file -
    else:
        # number of samples after the start of .nc file
        mm = int(t_in_nc / dt) 
        print('in .nc! mm = ', mm)
        # deal with edge problems - if the cand index is near the beginning
        if (mm<(window_width//2)):
            # then make the window size smaller 
            window_width = int(2*mm)
            if v==True:
                logging.info(f"Near the beginning, index = {mm}")
            # print('near the beginning')
        elif ((dur-mm)<window_width//2): # if near the end
            window_width = (dur-mm)
            if v==True:
                logging.info(f"Near the end, index = {mm}")
            # print('near the end', dur-mm)

    # put a warning if the window size is too small 
    if window_width < 10:
        print("Too few samples!")
        logging.warning(f"Window size too small! Will still proceed.")
        return
    print('mm = ', mm, 'windowsize = ', window_width)
    
    # dispersed candidate in xarray dataarray format.
    cand_disp = stokesi[(mm-window_width//2):(mm+window_width//2), :] 
    # write the dispersed pulse to a temporary .fil file
    ct.write_sigproc(fn_tempfil, cand_disp, t_start=T0+(mm-window_width//2)*dt/86400) 
    if v==True:
        logging.info(f"Done writing to a temporary .fil file.")
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
    if v==True:
        logging.info("Done reading the temporary .fil and dedispersing the candidate.")
    print('Done dedispersing')

    # generate a smaller window containing the dedispersed pulse and save to .fil
    # (before downsampling)
    window_time = 256 * tab['ibox'].values[0] # would like to keep 256 samples after downsampling
    # find the index of the pulse 
    mm = int((tab["mjds"].values[0] - cand.tstart)*86400 / dt)
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
                                    tstart=cand.tstart+(mm-window_width//2)*dt/86400,  
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
    if v==True:
        logging.info(f"Done saving the dedispersed pulse into filterbank file {fn_filout}.")

    # downsampling 
    # frequency ds by 16
    cand.decimate(key = 'ft', 
                  decimate_factor = 16, 
                  axis = 1) 
    # time ds by ibox
    cand.decimate(key = 'ft', 
                  decimate_factor = tab['ibox'].values[0],
                  axis = 0,
                  pad = True) 
    # time ds in DMtime domain
    cand.decimate(key='dmt',
                  decimate_factor = tab['ibox'].values[0],
                  axis = 1,
                  pad = True) 
    # update downsampled time resolution in cand.
    cand.tsamp = cand.tsamp * tab['ibox'].values[0]
    if v==True:
        logging.info(f"Done downsampling: cand.dedispersed.shape = {cand.dedispersed.shape}; cand.dmt.shape = {cand.dmt.shape}.")

    return(cand, tab)



def plot_grex(cand, tab, JSON): 

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
    
    window_time = 1024 # number of samples in the downsampled window
    ntime, nchans = cand.dedispersed.shape[0], cand.dedispersed.shape[1]
    f_low = int(278/16) # roughly from 1300MHz to 1500MHz, removing junks near the two edges. (in downsampled space)
    f_high = nchans - int(164/16) 
    
    cluster = pd.read_csv("/hdd/data/candidates/T2/cluster_output.csv")
    # snr,if,specnum,mjds,ibox,idm,dm,ibeam,cl,cntc,cntb,trigger
    this_cand = np.where(np.abs(cluster['mjds']-tab["mjds"].values[0])<30./86400)[0] # candidates nearby within 30s
    
    # seconds after the start of the intermediate .fil
    t_in_nc = (tab['mjds'].values[0] - cand.tstart) * 86400 
    # length of .fil file
    l = cand.dedispersed.shape[0] 
    if v==True:
        logging.info(f"Time in .nc = {t_in_nc}.")
        logging.info(f"Length of timestream = {l}.")
    
    if t_in_nc > l*cand.tsamp: # not captured in .fil, pick a random time interval
        logging.shutdown("Candidate not in the filterbank file. Shutting down.")
        return 
    # if in the .nc file
    else: 
        mm = int(t_in_nc / cand.tsamp)
        print('in .nc! mm = ', mm)
        if (mm<(window_time//2)):
            window_time = int(2*mm)
            if v==True:
                logging.info(f"In the intermediate.fil, near the beginning, index = {mm}")
        elif ((l-mm)<window_time//2):
            window_time = (l-mm)
            if v==True:
                logging.info(f"In the intermediate.fil, near the end, index = {mm}")

    print('mm, window = ', mm, window_time)

    if window_time<8:
        print('Too few samples to plot')
        logging.warning(f"Window size too small! Will still proceed.")
        return

    if v==True:
        logging.info(f"in plotting, mm = {mm}, time window = {window_time}")

    data_timestream = cand.dedispersed.mean(1)
    data_timestream = data_timestream[mm-window_time//2:mm+window_time//2]

    # Dedispersed pulse, remove channel mean
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
    tmin, tmax = times[0]-t_in_nc*1000, times[-1]-t_in_nc*1000
    # Construct the downsampled frequency array, after truncating highest and lowest edges
    freqs = np.linspace(cand.fch1+(nchans-f_low)*cand.foff*16, cand.fch1+cand.foff*16*(nchans-f_high), f_high-f_low)
    freqmin, freqmax = freqs[0], freqs[-1]

    # Calculate std of the given window.
    snr_tools = at.SNR_Tools()
    stds = snr_tools.calc_snr_presto(data_timestream, verbose=True)[1]
    
    # Plot
    logging.info("Starting to plot!")
    fig = plt.figure(figsize=(10,15))
    grid = plt.GridSpec(9, 6)

    # first row, collapse frequency -> time stream
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
               extent=(times.min()-t_in_nc*1000, times.max()-t_in_nc*1000, freqs.min(), freqs.max()),
               interpolation='nearest')
    DM0_delays = tmin + cand.dm * 4.15E6 * (freqmin**-2 - freqs**-2) # zero DM sweep
    plt.plot(DM0_delays, freqs, c='r', lw='2', alpha=0.35)
    plt.xlabel('Time (ms)+ MJD {}'.format(tab['mjds'].values[0]), fontsize=12)
    plt.ylabel('Frequency (MHz)', fontsize=12)
    plt.xlim(tmin, tmax)

    # DM vs. time
    plt.subplot(grid[4:6, 0:6])
    plt.imshow(data_dmt, 
               aspect='auto', 
               interpolation='nearest',
               extent=(times.min()-t_in_nc*1000, times.max()-t_in_nc*1000, 0, 2*cand.dm))
    plt.xlabel('Time (ms)+ MJD {}'.format(tab['mjds'].values[0]), fontsize=12)
    plt.ylabel(r'DM ($pc\cdot cm^{-3}$)', fontsize=12)
    
    # DM vs. MJD in cluster_output.csv
    plt.subplot(grid[7:9, 0:6])
    plt.scatter(cluster["mjds"][this_cand].values, 
                cluster["dm"][this_cand].values, 
                c=cluster['snr'][this_cand].values)
    plt.xlabel('Time (MJD)', fontsize=12)
    plt.ylabel(r'DM ($pc\cdot cm^{-3}$)', fontsize=12)

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

    logging.info("Done saving the plot.")
    
    plt.show()

    return()

### To run cand_plotter.py independently.
if __name__ == '__main__':
    c = js.split('.')[0] # candidate name 
    v = "/hdd/data/voltages/grex_dump-"+c+".nc" # voltage file
    fn_tempfil = "/hdd/data/candidates/T3/candplots/intermediate.fil" # output temporary .fil
    fn_outfil = f"/hdd/data/candidates/T3/cand{c}.fil" # output dedispersed candidate .fil
    (cand, tab) = gen_cand(v, fn_tempfil, fn_outfil, c+'.json')

    plot_grex(cand, tab, c+".json") 

    cmd = "rm {}".format(fn_tempfil)
    print(cmd)
    os.system(cmd)

