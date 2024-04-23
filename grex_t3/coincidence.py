import os
import sys

import time
import numpy as np
import matplotlib.pylab as plt
import glob
import pandas as pd 

from astropy.time import Time
import paramiko
import json
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from grex_t3 import analysis_tools 

fn_cluster_t2 = '/hdd/data/candidates/T2/cluster_output.csv'
fn_out_coincidence = '/hdd/data/candidates/T3/coincidence/coincidence.csv'
T2_dir = "/hdd/data/candidates/T2/"

class MyHandler(FileSystemEventHandler):
    """ This class enables us to watch the T2 directory for new JSON files
    and then coincidence them with the STARE candidates.
    """
    def on_created(self, event):
        # This function is called when a new file is created
        if event.is_directory:
            return
        if event.src_path.endswith('.json'):
            print(f'New JSON file detected: {event.src_path}')
            self.read_json(event.src_path)

    def read_json(self, file_path):
        # Function to read JSON file
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
        except Exception as e:
            print(f'Failed to read JSON file {file_path}: {e}')
            return

        self.coincidence_json(data)

    def coincidence_json(self,data):
        datadf = pd.DataFrame.from_dict(data, orient='index')
        # Reset the index to make it a column
        datadf.reset_index(inplace=True)
        # Name the new column
        datadf.rename(columns={'index': 'trigger'}, inplace=True)
        coincidence_array = coincidence_grex_stare(offset_utc_hours=7, t_thresh_sec=1.0, 
                           ncand_query=250, dm_diff_thresh=50.0, total_rows=0,
                           cands_grex=datadf)[0]
        
        if len(coincidence_array):
            print('Writing new coincidence to %s' % fn_out_coincidence)
            file_exists = os.path.isfile(fn_out_coincidence)
            coincidence_array.to_csv(fn_out_coincidence, mode='a' if file_exists else 'w', \
                                     header=not file_exists, index=False)

def check_for_newtrigger():
    # Set the directory you want to watch
    print("Starting coincidencing")
    observer = Observer()
    observer.schedule(MyHandler(), T2_dir, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

def fetch_external_cands(hostname='158.154.14.10', port=22, 
                        username='user', 
                        password='None', 
                        file_path='/home/user/cand_times_sync/heimdall_3.cand',
                        ncand=10):
    """Fetch the last N lines of a Heimdall candidate file on a remote server"""

    try:
        client.close()
    except:
        pass
    
    # Initialize the SSH client
    client = paramiko.SSHClient()
    
    # Add the server's SSH key automatically if it's missing
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        # Connect to the SSH server
        client.connect(hostname, port=port, username=username, password=password)
        
        # Command to fetch the last 10 lines of the file
        command = f"tail -n {ncand} {file_path}"
        
        # Execute the command
        stdin, stdout, stderr = client.exec_command(command)
        
        # Read the output from stdout
        lines = stdout.read()
        
        data_strings = lines.decode('utf-8').split('\n')
        # Split each string into a list of values
        data = [line.split() for line in data_strings if line.strip()]

        # Convert the list of lists into a DataFrame
        df = pd.DataFrame(data, index=None).astype(float)

        # Optionally, specify column names if known
        df.columns = ['snr','cand','time_sec',
                    'log2width','unknown2','dm',
                    'unknown3','mjdx','mjd_day',
                    'mjd_hr','mjd_min','mjd_sec']

        return df
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Close the connection
        client.close()

def coincidence_grex_stare(offset_utc_hours=7, t_thresh_sec=1.0, 
                           ncand_query=250, dm_diff_thresh=25., total_rows=0,
                           cands_grex=None):
    """Find coincidences between GREX and STARE candidates"""
    cands_stare = fetch_external_cands(hostname='158.154.14.10', port=22, 
                                        username='user', 
                                        password='None', 
                                        file_path='/home/user/cand_times_sync/heimdall_3.cand',
                                        ncand=ncand_query,
                                        )
    
    if cands_grex is None:
        # Step 1: Determine the number of rows in the file
        with open(fn_cluster_t2, 'r') as file:
            total_rows_now = sum(1 for row in file)

        if total_rows_now - total_rows < 10:
            print("Too few new candidates...")
            #return [], total_rows
        else:
            read_last_N = total_rows_now - total_rows

        # Step 2: Skip the first N-100 rows
        skip = max(1, total_rows_now - read_last_N)

        # Skip the first 1000 rows, but keep the Header
        cands_grex = pd.read_csv(fn_cluster_t2, skiprows=lambda x: x < skip and x != 0)
        cands_grex = cands_grex.iloc[np.where(cands_grex['trigger']!='0')[0]]

        if cands_grex.empty:
            print("No triggered candidates in GREX T2 data")
            return [], total_rows_now
    else:
        # If sending a single source from .json, assume total rows is 0
        total_rows_now = 0

    mjd_stare = analysis_tools.get_mjd_cand_pd(cands_stare, 
                                               offset_utc_hours=offset_utc_hours).values
    dms_stare = cands_stare['dm'].values
    candname_stare = cands_stare['cand'].values
    snr_stare = cands_stare['snr'].values

    coincidence_array = coincidence_grex_dumps(cands_grex, mjd_stare, dms_stare, candname_stare,
                                               snr_stare,
                                              t_thresh_sec=t_thresh_sec, 
                                              dm_diff_thresh=dm_diff_thresh)

    if len(coincidence_array)==0:
        print("No coincidences found")
    else:
        print("Found %d coincidences"%len(coincidence_array))
    
    return coincidence_array, total_rows_now

def run_coincidencer(fn_out_coincidence='/hdd/data/candidates/T3/coincidence/coincidence.csv',
                     t_thresh_sec=0.50,
                     offset_utc_hours=7.0,
                     dm_diff_thresh=25.0,):
    total_rows = 0

    while True:
        coincidence_array, total_rows = coincidence_grex_stare(t_thresh_sec=t_thresh_sec,
                                                   offset_utc_hours=offset_utc_hours,
                                                   dm_diff_thresh=dm_diff_thresh, 
                                                   total_rows=total_rows)
                
        if len(coincidence_array):
            file_exists = os.path.isfile(fn_out_coincidence)
            coincidence_array.to_csv(fn_out_coincidence, mode='a' if file_exists else 'w', \
                                     header=not file_exists, index=False)

            time.sleep(5.0)
        else:
            time.sleep(5.0)
            continue

def coincidence_grex_dumps(cands_grex, mjd_stare, dms_stare, candname_stare,
                           snr_stare,
                           t_thresh_sec=0.50, dm_diff_thresh=25.0):
    """ Find coincidences between GREX and STARE candidates.
    Step through each GREX trigger and find the STARE candidates 
    that are within t_thresh_sec and dm_diff_thresh of the GREX trigger.
    """
    mjd_grex = cands_grex['mjds']
    dms_grex = cands_grex['dm']
    delete_index = []

    for ii in range(len(mjd_grex)):
        locii = cands_grex.index[ii]
        delta_t_sec = np.abs(mjd_grex.iloc[ii]-mjd_stare)*86400
        delta_dm = np.abs(dms_grex.iloc[ii]-dms_stare)

        ind_coince = np.where((delta_t_sec<t_thresh_sec) & (delta_dm<dm_diff_thresh))[0]
        
        if len(ind_coince):
            jj = np.argmax(snr_stare[ind_coince])
            cands_grex.loc[locii, 'STAREMJD'] = mjd_stare[ind_coince[jj]]
            cands_grex.loc[locii, 'STAREDM'] = dms_stare[ind_coince[jj]]
            cands_grex.loc[locii, 'STARECAND'] = candname_stare[ind_coince[jj]]
            cands_grex.loc[locii, 'STARESNR'] = snr_stare[ind_coince[jj]]
        else:
            delete_index.append(locii)

    cands_grex_coincident = cands_grex.drop(delete_index)

    return cands_grex_coincident

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
    data_1 = analysis_tools.read_heim_pandas(fncand1, skiprows=0)
    data_2 = analysis_tools.read_heim_pandas(fncand2, skiprows=0)
    data_3 = analysis_tools.read_heim_pandas(fncand3, skiprows=0)

    mjd_arr_1 = analysis_tools.get_mjd_cand_pd(data_1).values
    mjd_arr_2 = analysis_tools.get_mjd_cand_pd(data_2).values

    try:
        mjd_arr_3 = analysis_tools.get_mjd_cand_pd(data_3).values
    except:
        print("Could not read fncand3 MJDs as pandas df")
        mjd_arr_3 = analysis_tools.get_mjd_cand(fncand3)

    # Find candidates that happened in past nday_lookback days
    ind1 = np.where((mjd_arr_1>(Time.now().mjd-nday_lookback)) & (mjd_arr_1<Time.now().mjd))[0]
    ind2 = np.where((mjd_arr_2>(Time.now().mjd-nday_lookback)) & (mjd_arr_1<Time.now().mjd))[0]
    ind3 = np.where((mjd_arr_3>(Time.now().mjd-nday_lookback)) & (mjd_arr_1<Time.now().mjd))[0]

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
    # try:
    #     nday_lookback = float(sys.argv[1])
    # except:
    #     nday_lookback = 1.

    # outdir = main(nday_lookback)
    check_for_newtrigger()




