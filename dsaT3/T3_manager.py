import traceback
import numpy as np
import glob
import subprocess
import time, os
import json
from dataclasses import asdict
from dask.distributed import Client, Lock

from dsautils import dsa_store
import dsautils.dsa_syslog as dsl
from event import event
from dsaT3 import filplot_funcs as filf
from dsaT3 import data_manager
from ovro_alert import alert_client


client = Client('10.42.0.232:8786')
LOCK = Lock('update_json')
ds = dsa_store.DsaStore()
LOGGER = dsl.DsaSyslogger()
LOGGER.subsystem("software")
LOGGER.app("dsaT3")
LOGGER.function("T3_manager")
dc = alert_client.AlertClient('dsa')

TIMEOUT_FIL = 600
FILPATH = '/dataz/dsa110/operations/T1/'
OUTPUT_PATH = '/dataz/dsa110/operations/T3/'
IP_GUANO = '3.13.26.235'

def submit_cand(fl, lock=LOCK):
    """ Given filename of trigger json, create DSACand and submit to scheduler for T3 processing.
    """

    d = event.create_event(fl)
    print(f"Submitting task for trigname {d.trigname}")

    d_fp = client.submit(run_filplot, d, key=f"run_filplot-{d.trigname}", wait=True, lock=lock, resources={'MEMORY': 10e9}, priority=-1)  # filplot and classify
    d_cs = client.submit(run_createstructure, d_fp, key=f"run_createstructure-{d.trigname}", lock=lock, priority=1)  # create directory structure
#    d_bf = client.submit(run_burstfit, d_fp, key=f"run_burstfit-{d.trigname}", lock=lock, priority=1)  # burstfit model fit
    d_vc = client.submit(run_voltagecopy, d_cs, key=f"run_voltagecopy-{d.trigname}", lock=lock)  # copy voltages
    d_h5 = client.submit(run_hdf5copy, d_cs, key=f"run_hdf5copy-{d.trigname}", lock=lock)  # copy hdf5
    d_fm = client.submit(run_fieldmscopy, d_cs, key=f"run_fieldmscopy-{d.trigname}", lock=lock)  # copy field image MS
#    d_hr = client.submit(run_hires, (d_bf, d_vc), key=f"run_hires-{d.trigname}", lock=lock)  # create high resolution filterbank
#    d_cm = client.submit(run_candidatems, (d_bf, d_vc), key=f"run_candidatems-{d.trigname}", lock=lock)  # make candidate image MS
#    d_po = client.submit(run_pol, d_hr, key=f"run_pol-{d.trigname}", lock=lock)  # run pol analysis on hires filterbank
#    d_hb = client.submit(run_hiresburstfit, d_hr, key=f"run_hiresburstfit-{d.trigname}", lock=lock)  # run burstfit on hires filterbank
#    d_il = client.submit(run_imloc, d_cm, key=f"run_imloc-{d.trigname}", lock=lock)  # run image localization on candidate image MS
#    d_as = client.submit(run_astrometry, (d_fm, d_cm), key=f"run_astrometry-{d.trigname}", lock=lock)  # astrometric burst image
#    fut = client.submit(run_final, (d_h5, d_po, d_hb, d_il, d_as), key=f"run_final-{d.trigname}", lock=lock)
    fut = client.submit(run_final, (d_h5, d_fm, d_vc), key=f"run_final-{d.trigname}", lock=lock)

    return fut


def run_filplot(d, wait=False, lock=None):
    """ Given DSACand, run filterbank analysis, plotting, and classification ("filplot").
    Returns DSACand with updated fields.
    """

    print('running filplot on {0}'.format(d.trigname))
    LOGGER.info('running filplot on {0}'.format(d.trigname))
    d.writejson(outpath=OUTPUT_PATH, lock=lock)

    ibeam = d.ibeam

    # TODO: get this from the dict set by T2, not from the name
    if '_inj' in d.trigname:
        d.injected = True
    else:
        d.injected = False

    if d.injected:
        print(f'Candidate {d.trigname} is an injection')
    else:
        print(f'Candidate {d.trigname} is not an injection')

    filfile = f"{FILPATH}/{d.trigname}/{d.trigname}_{ibeam}.fil"

    if wait:
        found_filfiles = wait_for_local_file(filfile, TIMEOUT_FIL, allbeams=True)
    else:
        found_filfiles = os.path.exists(filfile)

    if found_filfiles:
        d.filfile = filfile
    else:
        logging_string = 'Timeout while waiting for {0} filfiles'.format(d.trigname)
        logging_string += ' DM={0} ibox={1}'.format(d.dm, d.ibox)
        LOGGER.error(logging_string)
        filf.slack_client.chat_postMessage(channel='candidates', text=logging_string)
        d.candplot, d.probability, d.real = None, None, None
        return d

    # launch plot and classify
    try:
        # Test fast classifier:
        #d.ibeam_prob = filf.filplot_entry_fast(asdict(d), toslack=False, classify=True,
        #                        rficlean=False, ndm=1, nfreq_plot=32, save_data=False,
        #                        fllisting=None)

        d.candplot, d.probability, d.real = filf.filplot_entry(asdict(d), rficlean=False, classify=True)
        if d.probability > 0.95 and d.ibox < 16 and d.snr > 11 and not d.injected and (d.beams0 % 256 != d.beams1 % 256):
            # all injections lie on ns beam = ew beam (mod 256), so we are suspicious of those
            print("Running fast_response")
            fast_response(d)
        else:
            print('Not running fast_response. Event is too wide/weak/low-ibeam_prob/injection')

    except Exception as exception:
        logging_string = "Could not make filplot {0} due to {1}.  Callback:\n{2}".format(
            d.trigname,
            type(exception).__name__,
            ''.join(
                traceback.format_tb(exception.__traceback__)
            )
        )
        print(logging_string)
        LOGGER.error(logging_string)

        d.candplot, d.probability, d.real = None, None, None
        filf.slack_client.chat_postMessage(channel='candidates', text=logging_string)

    d.writejson(outpath=OUTPUT_PATH, lock=lock)
    
    return d


def fast_response(d):
    """ Use DSACand with fast classification to do fast response (e.g., set relay or send VOEvent)
    """

    infile = os.path.join(OUTPUT_PATH, d.trigname + '.json')
    outfile = os.path.join(OUTPUT_PATH, d.trigname + '.xml')

    if not os.path.exists(infile):
        print(f"{infile} not found. Waiting for it to appear...")
        elapsed = 0
        waitloop = 5
        while not os.path.exists(infile):
            print(f"Not found yet...")
            time.sleep(waitloop)
            elapsed += waitloop
            if elapsed > TIMEOUT_FIL:
                print(f"Giving up on {infile}.")
                break

    ret = 1
    if os.path.exists(infile):
        ret = subprocess.run(['dsaevent', 'create-voevent', infile, outfile, '--production']).returncode
        if not d.injected:
            dc.set('observation', args=asdict(d))
            if ret == 0:
                print(f"Non-injection VOEvent created, but NOT sending {outfile}...")
                filf.slack_client.chat_postMessage(channel='candidates', text=f'NOT sending VOEvent {outfile}...')
# commented out for testing                
#                print(f"Non-injection VOEvent created. Sending {outfile}...")
#                ret = subprocess.run(['dsaevent', 'send-voevent', '--destination', IP_GUANO, outfile]).returncode
#                filf.slack_client.chat_postMessage(channel='candidates', text=f'Sending VOEvent {outfile}...')
            else:
                print(f"Non-injection event, but VOEvent {outfile} not created...")
        else:
            dc.set('test', args=asdict(d))
    else:
        print(f"Could not find {infile}, so no {outfile} made or event sent.")

    # TODO: is this ASAP with updated position later? or wait to send with good position?


def run_createstructure(d, lock=None):
    """ Use DSACand (after filplot) to decide on creating/copying files to candidate data area.
    """

    if d.real and not d.injected:
        print("Running createstructure for real/non-injection candidate.")

        # TODO: have DataManager parse DSACand
        dm = data_manager.DataManager(d.__dict__)
        # TODO: have update method accept dict or DSACand
        d.update(dm())

    else:
        print("Not running createstructure for non-astrophysical candidate.")

    d.writejson(outpath=OUTPUT_PATH, lock=lock)
    return d


def run_burstfit(d, lock=None):
    """ Given DSACand, run burstfit analysis.
    Returns new dictionary with refined DM, width, arrival time.
    """

    from burstfit.BurstFit_paper_template import real_time_burstfit

    if d.real:
        print('Running burstfit on {0}'.format(d.trigname))
        LOGGER.info('Running burstfit on {0}'.format(d.trigname))
        d_bf = real_time_burstfit(d.trigname, d.filfile, d.snr, d.dm, d.ibox)

        d.update(d_bf)
        d.writejson(outpath=OUTPUT_PATH, lock=lock)
    else:
        print('Not running burstfit on {0}'.format(d.trigname))
        LOGGER.info('Not running burstfit on {0}'.format(d.trigname))

    return d


def run_hdf5copy(d, lock=None):
    """ Given DSACand (after filplot), copy hdf5 files
    """

    if d.real and not d.injected:
        print('Running hdf5copy on {0}'.format(d.trigname))
        LOGGER.info('Running hdf5copy on {0}'.format(d.trigname))

        dm = data_manager.DataManager(d.__dict__)
        dm.link_hdf5_files()

        d.update(dm.candparams)
        d.writejson(outpath=OUTPUT_PATH, lock=lock)

    return d


def run_voltagecopy(d, lock=None):
    """ Given DSACand (after filplot), copy voltage files.
    """

    if d.real and not d.injected:    
        print('Running voltagecopy on {0}'.format(d.trigname))
        LOGGER.info('Running voltagecopy on {0}'.format(d.trigname))
        dm = data_manager.DataManager(d.__dict__)
        dm.copy_voltages()

        # TODO: have update method accept dict or DSACand
        d.update(dm.candparams)
        d.writejson(outpath=OUTPUT_PATH, lock=lock)

    return d


def run_hires(ds, lock=None):
    """ Given DSACand objects from burstfit and voltage, generate hires filterbank files.
    """

    d, d_vc = ds
    d.update(d_vc)

    print('placeholder run_hires on {0}'.format(d.trigname))
    LOGGER.info('placeholder run_hires on {0}'.format(d.trigname))

#    if dd['real'] and not dd['injected']:
    d.writejson(outpath=OUTPUT_PATH, lock=lock)

    return d


def run_pol(d, lock=None):
    """ Given DSACand (after hires), run polarization analysis.
    Returns updated DSACand with new file locations?
    """

    print('placeholder nrun_pol on {0}'.format(d.trigname))
    LOGGER.info('placeholder run_pol on {0}'.format(d.trigname))

#    if d_hr['real'] and not d_hr['injected']:
    d.writejson(outpath=OUTPUT_PATH, lock=lock)

    return d


def run_fieldmscopy(d, lock=None):
    """ Given DSACand (after filplot), copy field MS file.
    Returns updated DSACand with new file locations.
    """

    print('placeholder run_fieldmscopy on {0}'.format(d.trigname))
    LOGGER.info('placeholder run_fieldmscopy on {0}'.format(d.trigname))

#    if d_fp['real'] and not d_fp['injected']:
#        dm = data_manager.DataManager(d_fp)
#        dm.link_field_ms()
#        update_json(dm.candparams, lock=lock)
#        return dm.candparams
#    else:
    return d


def run_candidatems(ds, lock=None):
    """ Given DSACands from filplot and voltage copy, make candidate MS image.
    Returns updated DSACand with new file locations.
    """

    d, d_vc = ds
    d.update(d_vc)

    print('placeholder run_candidatems on {0}'.format(d.trigname))
    LOGGER.info('placeholder run_candidatems on {0}'.format(d.trigname))

#    if dd['real'] and not dd['injected']:

    d.writejson(outpath=OUTPUT_PATH, lock=lock)

    return d


def run_hiresburstfit(d, lock=None):
    """ Given DSACand, run highres burstfit analysis.
    Returns updated DSACand with new file locations.
    """

    print('placeholder run_hiresburstfit on {0}'.format(d.trigname))
    LOGGER.info('placeholder run_hiresburstfit on {0}'.format(d.trigname))

#    if d_hr['real'] and not d_hr['injected']:
    d.writejson(outpath=OUTPUT_PATH, lock=lock)

    return d


def run_imloc(d, lock=None):
    """ Given DSACand (after candidate image MS), run image localization.
    """

    print(f'Running localization on {d.trigname}')
    LOGGER.info(f'Running localization on {d.trigname}')

# TODO: is this the first sent or an update with good position?
#    if d.real and not d.injected:
#        dc.set('observation', args=asdict(d))

    d.writejson(outpath=OUTPUT_PATH, lock=lock)
    return d


def run_astrometry(ds, lock=None):
    """ Given field image MS and candidate image MS, run astrometric localization analysis.
    """

    d, d_cm = ds
    d.update(d_cm)

    print('placeholder run_astrometry on {0}'.format(d.trigname))
    LOGGER.info('placeholder run_astrometry on {0}'.format(d.trigname))

#    if dd['real'] and not dd['injected']:

    d.writejson(outpath=OUTPUT_PATH, lock=lock)

    return d


def run_final(ds, lock=None):
    """ Reduction task to handle all final tasks in graph.
    May also update etcd to notify of completion.
    """

#    d, d_po, d_hb, d_il, d_as = ds
    d, d_fm, d_vc = ds
    d.update(d_fm)
    d.update(d_vc)
#    d.update(d_il)
#    d.update(d_as)

    print('Final merge of results for {0}'.format(d.trigname))
    LOGGER.info('Final merge of results for {0}'.format(d.trigname))

    d.writejson(outpath=OUTPUT_PATH, lock=lock)

    return d


def wait_for_local_file(fl, timeout, allbeams=False):
    """ Wait for file named fl to be written. fl can be string filename of list of filenames.
    If timeout (in seconds) exceeded, then return None.
    allbeams will parse input (str) file name to get list of all beam file names.
    """

    if allbeams:
        assert isinstance(fl, str), 'Input should be detection beam fil file'
        loc = os.path.dirname(fl)
        fl0 = os.path.basename(fl.rstrip('.fil'))
        fl1 = "_".join(fl0.split("_")[:-1])
        fl = [f"{os.path.join(loc, fl1 + '_' + str(i) + '.fil')}" for i in range(512)]
    
    if isinstance(fl, str):
        fl = [fl]
    assert isinstance(fl, list), "name or list of fil files expected"

    elapsed = 0
    while not all([os.path.exists(ff) for ff in fl]):
        time.sleep(5)
        elapsed += 5
        if elapsed > timeout:
            return None
        elif elapsed <= 5:
            print(f"Waiting for {len(fl)} files, like {fl[0]}...")

    return fl
