import traceback
import numpy as np
import glob
from dsautils import dsa_store
import dsautils.dsa_syslog as dsl
from dsaT3 import filplot_funcs as filf
from dsaT3 import data_manager
import time, os
import json
from dask.distributed import Client

client = Client('10.42.0.232:8786')
ds = dsa_store.DsaStore()
LOGGER = dsl.DsaSyslogger()
LOGGER.subsystem("software")
LOGGER.app("dsaT3")
LOGGER.function("T3_manager")

TIMEOUT_FIL = 600
FILPATH = '/dataz/dsa110/operations/T1/'
OUTPUT_PATH = '/dataz/dsa110/operations/T3/'


# TODO: change all run_* functions to take DSAEvent as input
def run_filplot(a, wait=False, lock=None):
    """ Given candidate dictionary, run filterbank analysis, plotting, and classification ("filplot").
    Returns dictionary with updated fields.
    """

    # set up output dict and datestring
    output_dict = a.copy()
    fill_empty_dict(output_dict)

    print('running filplot on {0}'.format(output_dict['trigname']))
    LOGGER.info('running filplot on {0}'.format(output_dict['trigname']))

    # wait for specific filterbank file to be written
    ibeam = output_dict['ibeam'] + 1
    trigname = output_dict['trigname']

    # TODO: get this from the dict set by T2, not from the name
    if 'injected' not in output_dict:
        if '_inj' in trigname:
            output_dict['injected'] = True
        else:
            output_dict['injected'] = False

    if output_dict['injected']:
        print(f'Candidate {trigname} is an injection')
    else:
        print(f'Candidate {trigname} is not an injection')

    filfile = f"{FILPATH}/{trigname}/{trigname}_{ibeam}.fil"

    if wait:
        found_filfiles = wait_for_local_file(filfile, TIMEOUT_FIL, allbeams=True)
    else:
        found_filfiles = os.path.exists(filfile)

    if found_filfiles:
        output_dict['filfile'] = filfile
    else:
        logging_string = 'Timeout while waiting for {0} filfiles'.format(output_dict['trigname'])
        LOGGER.error(logging_string)
        filf.slack_client.chat_postMessage(channel='candidates', text=logging_string)
        output_dict['candplot'], output_dict['probability'], output_dict['real'] = None, None, None
        return output_dict
    
    # launch plot and classify
    try:
        output_dict['candplot'], output_dict['probability'], output_dict['real'] = filf.filplot_entry(output_dict, rficlean=False)
    except Exception as exception:
        logging_string = "Could not make filplot {0} due to {1}.  Callback:\n{2}".format(
            output_dict['trigname'],
            type(exception).__name__,
            ''.join(
                traceback.format_tb(exception.__traceback__)
            )
        )
        print(logging_string)
        LOGGER.error(logging_string)

        output_dict['candplot'], output_dict['probability'], output_dict['real'] = None, None, None
        filf.slack_client.chat_postMessage(channel='candidates', text=logging_string)

    update_json(output_dict, lock=lock)
    
    return output_dict


def run_createstructure(dd, lock=None):
    """ Given filplot results, decide on creating/copying files to candidate data area.
    """

    if dd['real'] and not dd['injected']:
        print("Running createstructure for real/non-injection candidate.")
        dm = data_manager.DataManager(dd)
        dd = dm()
    else:
        print("Not running createstructure for non-astrophysical candidate.")

    update_json(dd, lock=lock)
    return dd


def run_burstfit(dd, lock=None):
    """ Given candidate dictionary, run burstfit analysis.
    Returns new dictionary with refined DM, width, arrival time.
    """

    from burstfit.BurstFit_paper_template import real_time_burstfit

    if dd['real']:
        print('Running burstfit on {0}'.format(dd['trigname']))
        LOGGER.info('Running burstfit on {0}'.format(dd['trigname']))
        d_bf = real_time_burstfit(dd['trigname'], dd['filfile'], dd['snr'], dd['dm'], dd['ibox'])
        dd.update(d_bf)
        update_json(dd, lock=lock)
    else:
        print('Not running burstfit on {0}'.format(dd['trigname']))
        LOGGER.info('Not running burstfit on {0}'.format(dd['trigname']))

    return dd.copy()


def run_hdf5copy(d_fp, lock=None):
    """ Given filplot candidate dictionary, copy hdf5 files
    """

    if d_fp['real'] and not d_fp['injected']:
        print('Running hdf5copy on {0}'.format(d_fp['trigname']))
        LOGGER.info('Running hdf5copy on {0}'.format(d_fp['trigname']))

        dm = data_manager.DataManager(d_fp)
        dm.link_hdf5_files()
        update_json(dm.candparams, lock=lock)
        return dm.candparams
    else:
        return d_fp.copy()


def run_voltagecopy(d_fp, lock=None):
    """ Given filplot candidate dictionary, copy voltage files
    """

    if d_fp['real'] and not d_fp['injected']:    
        print('Running voltagecopy on {0}'.format(d_fp['trigname']))
        LOGGER.info('Running voltagecopy on {0}'.format(d_fp['trigname']))
        dm = data_manager.DataManager(d_fp)
        dm.copy_voltages()
        update_json(dm.candparams, lock=lock)
        return dm.candparams
    else:
        return d_fp


def run_hires(dds, lock=None):
    """ Given burstfit and voltage dictionaries, generate hires filterbank files.
    """

    d_bf, d_vc = dds
    dd = d_bf.copy()

    print('placeholder run_hires on {0}'.format(dd['trigname']))
    LOGGER.info('placeholder run_hires on {0}'.format(dd['trigname']))

#    if dd['real'] and not dd['injected']:
    dd.update(d_vc)

    update_json(dd, lock=lock)

    return dd


def run_pol(d_hr, lock=None):
    """ Given hires candidate dictionary, run polarization analysis.
    Returns new dictionary with new file locations?
    """

    print('placeholder nrun_pol on {0}'.format(d_hr['trigname']))
    LOGGER.info('placeholder run_pol on {0}'.format(d_hr['trigname']))

#    if d_hr['real'] and not d_hr['injected']:
    update_json(d_hr, lock=lock)

    return d_hr.copy()


def run_fieldmscopy(d_fp, lock=None):
    """ Given filplot candidate dictionary, copy field MS file.
    Returns new dictionary with new file locations.
    """

    print('placeholder run_fieldmscopy on {0}'.format(d_fp['trigname']))
    LOGGER.info('placeholder run_fieldmscopy on {0}'.format(d_fp['trigname']))

#    if d_fp['real'] and not d_fp['injected']:
#        dm = data_manager.DataManager(d_fp)
#        dm.link_field_ms()
#        update_json(dm.candparams, lock=lock)
#        return dm.candparams
#    else:
    return d_fp.copy()


def run_candidatems(dds, lock=None):
    """ Given filplot and voltage copy candidate dictionaries, make candidate MS image.
    Returns new dictionary with new file locations.
    """

    d_bf, d_vc = dds
    dd = d_bf.copy()

    print('placeholder run_candidatems on {0}'.format(dd['trigname']))
    LOGGER.info('placeholder run_candidatems on {0}'.format(dd['trigname']))

#    if dd['real'] and not dd['injected']:
    dd.update(d_vc)

    update_json(dd, lock=lock)

    return dd


def run_hiresburstfit(d_hr, lock=None):
    """ Given hires candidate dictionary, run highres burstfit analysis.
    Returns new dictionary with new file locations.
    """

    print('placeholder run_hiresburstfit on {0}'.format(d_hr['trigname']))
    LOGGER.info('placeholder run_hiresburstfit on {0}'.format(d_hr['trigname']))

#    if d_hr['real'] and not d_hr['injected']:
    update_json(d_hr, lock=lock)

    return d_hr.copy()


def run_imloc(d_cm, lock=None):
    """ Given candidate image MS, run image localization.
    """

    print('placeholder run_imloc on {0}'.format(d_cm['trigname']))
    LOGGER.info('placeholder run_imloc on {0}'.format(d_cm['trigname']))

#    if d_cm['real'] and not d_cm['injected']:
    update_json(d_cm, lock=lock)

    return d_cm.copy()


def run_astrometry(dds, lock=None):
    """ Given field image MS and candidate image MS, run astrometric localization analysis.
    """

    d_fm, d_cm = dds
    dd = d_fm.copy()

    print('placeholder run_astrometry on {0}'.format(dd['trigname']))
    LOGGER.info('placeholder run_astrometry on {0}'.format(dd['trigname']))

#    if dd['real'] and not dd['injected']:
    dd.update(d_cm)

    update_json(dd, lock=lock)

    return dd


def run_final(dds, lock=None):
    """ Token task to handle all final tasks in graph.
    May also update etcd to notify of completion.
    """

    d_h5, d_po, d_hb, d_il, d_as = dds
    dd = d_h5.copy()

    print('Final merge of results for {0}'.format(dd['trigname']))
    LOGGER.info('Final merge of results for {0}'.format(dd['trigname']))

    dd.update(d_po)
    dd.update(d_hb)
    dd.update(d_il)

    update_json(dd, lock=lock)

    return dd

# TODO: this may become method of DSAEvent
def update_json(dd, lock=None, outpath=OUTPUT_PATH):
    """ Lock, read, write, unlock json file on disk.
    Uses trigname field to find file
    """

    fn = outpath + dd['trigname'] + '.json'

    if lock is not None:
        lock.acquire(timeout="5s")
    
    if not os.path.exists(fn):
        with open(fn, 'w') as f:
            json.dump(dd, f, ensure_ascii=False, indent=4)
    else:
        try:
            with open(fn, 'r') as f:
                extant_json = json.load(f)
                extant_json.update(dd)
                with open(fn, 'w') as f:
                    json.dump(extant_json, f, ensure_ascii=False, indent=4)
        except json.JSONDecodeError:
            with open(fn, 'w') as f:
                json.dump(dd, f, ensure_ascii=False, indent=4)

    if lock is not None:
        lock.release()


# TODO: this may be obsolete after DSAEvent restructuring
def fill_empty_dict(od, emptyCorrs=True, correctCorrs=False):
    """ Takes standard candidate dict, od, and resets entries to default values (e.g., None/False).
    """

    od['filfile'] = None
    od['candplot'] = None
    od['save'] = False
    od['label'] = None
    if emptyCorrs is True:
        for corr in ['corr03','corr04','corr05','corr06','corr07','corr08','corr10','corr11','corr12','corr14','corr15','corr16','corr18','corr19','corr21','corr22']:
            od[corr+'_data'] = None
            od[corr+'_header'] = None

    if correctCorrs is True:
        for corr in ['corr03','corr04','corr05','corr06','corr07','corr08','corr10','corr11','corr12','corr14','corr15','corr16','corr18','corr19','corr21','corr22']:
            if od[corr+'_data'] is not None:
                od[corr+'_data'] = od[corr+'_data'][:-19]
            if od[corr+'_header'] is not None:
                od[corr+'_header'] = od[corr+'_header'][:-22]
        

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
