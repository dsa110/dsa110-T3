"""
A script to convert voltage files to measurement sets.
"""
import json
import re
import os
import glob
import subprocess
from multiprocessing import Process, Manager
import multiprocessing
import queue
import argparse
import time
import yaml
from pkg_resources import resource_filename
from astropy.time import Time
import astropy.units as u
from dsaT3.utils import get_declination_mjd, rsync_file
from dsaT3.T3imaging import generate_T3_uvh5
from dsacalib.ms_io import uvh5_to_ms
from dsautils import cnf

NPROC = 8
PARAMFILE = resource_filename('dsaT3', 'data/T3_parameters.yaml')
with open(PARAMFILE) as YAMLF:
    T3PARAMS = yaml.load(YAMLF, Loader=yaml.FullLoader)['T3corr']
CONF = cnf.Conf()
CORR_LIST = list(CONF.get('corr')['ch0'].keys())

def rsync_handler(
        rsync_queue,
        corr_queue,
        rsync,
):
    rsync_done = False
    while not rsync_done:
        try:
            item = rsync_queue.get()
        except queue.Empty:
            time.sleep(10)
        else:
            if item == 'END':
                rsync_done = True
                continue
            srcfile, vfile = item
            if not os.path.exists(vfile):
                if rsync:
                    rsync_file(
                        srcfile,
                        vfile
                    )
                else:
                    os.symlink(srcfile, vfile)
            corr_queue.put(vfile)

def corr_handler(
        deltat_ms,
        deltaf_MHz,
        corr_queue,
        uvh5_queue,
        ncorrfiles,
        ncorrfiles_lock
):
    """Correlates data using T3 cpu correlator.

    Parameters
    ----------
    deltat_ms : float
        The desired integration time in the correlated data, in ms.
    deltaf_MHz : float
        The desired integration frequency in the correlated data, in MHz.
    """
    corr_done = False
    while not corr_done:
        if ncorrfiles.value > NPROC-1:
            time.sleep(10)
            continue
        try:
            vfile = corr_queue.get()
        except queue.Empty:
            time.sleep(10)
        else:
            if vfile == 'END':
                corr_done = True
                continue
            with ncorrfiles_lock:
                ncorrfiles.value += 1
            if not os.path.exists('{0}.corr'.format(vfile)):
                command = (
                    '/home/ubuntu/proj/dsa110-shell/dsa110-bbproc/dsacorr '
                    '-d {0} -o {0}.corr -t {1} -f {2} -a {3}'.format(
                        vfile,
                        deltat_ms,
                        deltaf_MHz, 
                        len(T3PARAMS['antennas'])
                    )
                )
                print(command)
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    shell=True
                )
                proc_stdout = str(process.communicate()[0].strip())
                print(proc_stdout)
            corr_files = dict({})
            corr = re.findall('corr\d\d', vfile)[0]
            corr_files[corr] = '{0}.corr'.format(vfile)
            uvh5_queue.put(corr_files)

def uvh5_handler(
        candname,
        declination,
        tstart,
        ntint,
        nfint,
        start_offset,
        end_offset,
        uvh5_queue,
        ncorrfiles,
        ncorrfiles_lock
):
    proc = multiprocessing.current_process()
    uvh5_done = False
    while not uvh5_done:
        try:
            corr_files = uvh5_queue.get()
        except queue.Empty:
            time.sleep(10)
        else:
            if corr_files == 'END':
                print('proc {0} is setting uvh5_done'.format(proc.pid))
                uvh5_done = True
                continue
            uvh5name = generate_T3_uvh5(
                '{0}/{1}'.format(T3PARAMS['corrdir'], candname),
                declination,
                tstart,
                ntint=ntint,
                nfint=nfint,
                filelist=corr_files,
                start_offset=start_offset,
                end_offset=end_offset
            )
            print(uvh5name)
            #for value in corr_files.values():
            #    os.remove(value)
            with ncorrfiles_lock:
                ncorrfiles.value -= 1
    print('{0} exiting'.format(proc.pid))

def __main__(candname, datestring, ntint, nfint, start_offset, end_offset):
    """
    Correlate voltage files and convert to a measurement set.

    Parameters
    ----------
    candname : str
        The unique name of the candidate.
    datestring : str
        The datestring the observation is archived under. Use 'current' if the
        data is from the current, unarchived observing run.
    filelist : list
        The full paths to the voltage files on dsa-storage.
    ntint : int
        The number of time samples to integrate together during correlation.
    nfint : int
        The number of frequency channels to integrate together after removing
        outrigger delays.
    start_offset : int
        The number of time samples (after correlation) to offset the start of
        the measurement set by.  If not provided, the entire time is converted
        to a measurement set.
    end_offset : int
        The last time sample (after correlation) to write to the measurement
        set. If not provide,d the entire time is converted to a measurement
        set.
    """
    if start_offset < 0:
        start_offset = None
    if end_offset < 0:
        end_offset = None
    if datestring == 'current':
        rsync = True
        filenames = [
            '{0}.sas.pvt:/home/ubuntu/data/{1}_data.out'.format(
                corr,
                candname
            ) for corr in CORR_LIST
        ]
        headername = '{0}/{1}.json'.format(T3PARAMS['T3dir'], candname)
    else:
        rsync = False
        filenames = [
            '{0}/{1}/{2}_{3}_data.out'.format(
                T3PARAMS['archivedir'],
                datestring,
                corr,
                candname
            ) for corr in CORR_LIST
        ]
        headername = '{0}/{1}/{2}.json'.format(
            T3PARAMS['archivedir'],
            datestring,
            candname
        )
    outnames = [
        '{0}/{1}_{2}_data.out'.format(
            T3PARAMS['corrdir'],
            corr,
            candname
        ) for corr in CORR_LIST
    ]
    # Get metadata
    with open(headername) as jsonf:
        metadata = json.load(jsonf)
    tstart = Time(metadata['mjds'], format='mjd')
    try:
        declination = get_declination_mjd(tstart)
    except (ConnectionError, KeyError):
        declination = 54.58209895*u.deg
    deltat_ms = ntint*T3PARAMS['deltat_s']*1e3
    deltaf_MHz = T3PARAMS['deltaf_MHz']

    manager = Manager()
    ncorrfiles = manager.Value('i', 0)
    ncorrfiles_lock = manager.Lock()
    rsync_queue = manager.Queue()
    corr_queue = manager.Queue()
    uvh5_queue = manager.Queue()
    # Copy files
    for i, filename in enumerate(filenames):
        rsync_queue.put([filename, outnames[i]])
    rsync_queue.put('END')
    processes = []
    processes += [Process(
        target=rsync_handler,
        args=(
            rsync_queue,
            corr_queue,
            rsync
        ),
        daemon=True
    )]
    for i in range(NPROC):
        processes += [Process(
            target=corr_handler,
            args=(
                deltat_ms,
                deltaf_MHz,
                corr_queue,
                uvh5_queue,
                ncorrfiles,
                ncorrfiles_lock
            ),
            daemon=True
        )]
    for i in range(NPROC):
        processes += [Process(
            target=uvh5_handler,
            args=(
                candname,
                declination,
                tstart,
                ntint,
                nfint,
                start_offset,
                end_offset,
                uvh5_queue,
                ncorrfiles,
                ncorrfiles_lock
            ),
            daemon=True
        )]
    for proc in processes:
        proc.start()
    processes[0].join()
    for i in range(NPROC):
        corr_queue.put('END')
    for proc in processes[1:NPROC+1]:
        proc.join()
    print('All corr processes done')
    # We get here
    for i in range(NPROC):
        uvh5_queue.put('END')
    for proc in processes[1+NPROC:]:
        proc.join()
        print('A uvh5 process joined.')
    print('All uvh5 processes done')
    hdf5files = sorted(glob.glob('{0}/{1}_corr??.hdf5'.format(
        T3PARAMS['corrdir'],
        candname
    )))
    uvh5_to_ms(
        hdf5files,
        '{0}/{1}'.format(T3PARAMS['msdir'], candname)
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Correlate candidate voltage files.'
    )
    parser.add_argument(
        'candname',
        type=str,
        help='unique candidate name'
    )
    parser.add_argument(
        '--datestring',
        type=str,
        help='datestring of archived candidate',
        nargs='?',
        default='current'
    )
    parser.add_argument(
        '--ntint',
        type=int,
        nargs='?',
        default=8,
        help='number of native time bins to integrate during correlation'
    )
    parser.add_argument(
        '--nfint',
        type=int,
        nargs='?',
        default=8,
        help='number of native freq bins to integrate during correlation'
    )
    parser.add_argument(
        '--startoffset',
        type=int,
        nargs='?',
        default=1716,
        help='nbins from beginning of correlated data to start writing to ms'
    )
    parser.add_argument(
        '--stopoffset',
        type=int,
        nargs='?',
        default=2484,
        help='number of bins from end of correlation to write to ms'
    )
    args = parser.parse_args()
    __main__(args.candname, args.datestring, ntint=args.ntint, nfint=args.nfint,
             start_offset=args.startoffset, end_offset=args.stopoffset)
