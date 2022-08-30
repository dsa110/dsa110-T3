import traceback
import numpy as np
import glob
from dsautils import dsa_store
import dsautils.dsa_syslog as dsl
from event import event
from dsaT3 import filplot_funcs as filf
from dsaT3 import data_manager
import time, os
import json
from dask.distributed import Client, Lock

client = Client('10.42.0.232:8786')
LOCK = Lock('update_json')
ds = dsa_store.DsaStore()
LOGGER = dsl.DsaSyslogger()
LOGGER.subsystem("software")
LOGGER.app("dsaT3")
LOGGER.function("T3_manager")

TIMEOUT_FIL = 600
FILPATH = '/dataz/dsa110/operations/T1/'
OUTPUT_PATH = '/dataz/dsa110/operations/T3/'


def submit_cand(fl, LOCK=None):
    """ Given filename of trigger json, create DSACand and submit to scheduler for T3 processing.
    """

    d = event.create_event(fl)
    print(f"Submitting task for trigname {d.trigname}")

    d_fp = client.submit(run_filplot, d, key=f"run_filplot-{d.trigname}", wait=True, lock=LOCK, resources={'MEMORY': 10e9}, priority=-1)  # filplot and classify
    d_cs = client.submit(run_createstructure, d_fp, key=f"run_createstructure-{d.trigname}", lock=LOCK, priority=1)  # create directory structure
    d_bf = client.submit(run_burstfit, d_fp, key=f"run_burstfit-{d.trigname}", lock=LOCK, priority=1)  # burstfit model fit
    d_vc = client.submit(run_voltagecopy, d_cs, key=f"run_voltagecopy-{d.trigname}", lock=LOCK)  # copy voltages
    d_h5 = client.submit(run_hdf5copy, d_cs, key=f"run_hdf5copy-{d.trigname}", lock=LOCK)  # copy hdf5
    d_fm = client.submit(run_fieldmscopy, d_cs, key=f"run_fieldmscopy-{d.trigname}", lock=LOCK)  # copy field image MS
    d_hr = client.submit(run_hires, (d_bf, d_vc), key=f"run_hires-{d.trigname}", lock=LOCK)  # create high resolution filterbank
    d_cm = client.submit(run_candidatems, (d_bf, d_vc), key=f"run_candidatems-{d.trigname}", lock=LOCK)  # make candidate image MS
    d_po = client.submit(run_pol, d_hr, key=f"run_pol-{d.trigname}", lock=LOCK)  # run pol analysis on hires filterbank
    d_hb = client.submit(run_hiresburstfit, d_hr, key=f"run_hiresburstfit-{d.trigname}", lock=LOCK)  # run burstfit on hires filterbank
    d_il = client.submit(run_imloc, d_cm, key=f"run_imloc-{d.trigname}", lock=LOCK)  # run image localization on candidate image MS
    d_as = client.submit(run_astrometry, (d_fm, d_cm), key=f"run_astrometry-{d.trigname}", lock=LOCK)  # astrometric burst image
    fut = client.submit(run_final, (d_h5, d_po, d_hb, d_il, d_as), key=f"run_final-{d.trigname}", lock=LOCK)

    return fut


def run_filplot(d, wait=False, lock=None):
    """ Given DSACand, run filterbank analysis, plotting, and classification ("filplot").
    Returns DSACand with updated fields.
    """

    print('running filplot on {0}'.format(d.trigname))
    LOGGER.info('running filplot on {0}'.format(d.trigname))

    ibeam = d.ibeam + 1

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
        LOGGER.error(logging_string)
        filf.slack_client.chat_postMessage(channel='candidates', text=logging_string)
        d.candplot, d.probability, d.real = None, None, None
        return d
    
    # launch plot and classify
    try:
        # TODO: filplot can handle DSACand
        d.candplot, d.probability, d.real = filf.filplot_entry(d.__dict__, rficlean=False)
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


def run_createstructure(d, lock=None):
    """ Use DSACand (after filplot) to decide on creating/copying files to candidate data area.
    """

    if d.real and not d.injected:
        print("Running createstructure for real/non-injection candidate.")

        # TODO: have DataManager parse DSACand
        dm = data_manager.DataManager(d.__dict__)
        # TODO: have update method accept dict or DSACand
        d.__dict__.update(dm())

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

        # TODO: have update method accept dict or DSACand
        d.__dict__.update(d_bf)

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

        # TODO: have update method accept dict or DSACand
        d.__dict__.update(dm.candparams)
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
        d.__dict__.update(dm.candparams)
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

    print('placeholder run_imloc on {0}'.format(d.trigname))
    LOGGER.info('placeholder run_imloc on {0}'.format(d.trigname))

#    if d_cm['real'] and not d_cm['injected']:

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

    d, d_po, d_hb, d_il, d_as = ds
    d.update(d_po)
    d.update(d_hb)
    d.update(d_il)
    d.update(d_as)

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
        fl = [f"{os.path.join(loc, fl1 + '_' + str(i) + '.fil')}" for i in range(256)]
    
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
            print(f"Waiting for files {fl}...")

    return fl
