import traceback
import numpy as np
from dsautils import dsa_store
import dsautils.dsa_syslog as dsl
from dsaT3 import filplot_funcs as filf
import time, os
import json
from dask.distributed import Lock

ds = dsa_store.DsaStore()
LOCK = Lock()
LOGGER = dsl.DsaSyslogger()
LOGGER.subsystem("software")
LOGGER.app("dsaT3")
LOGGER.function("T3_manager")

TIMEOUT_FIL = 60
FILPATH = '/dataz/dsa110/operations/T1/'
OUTPUT_PATH = '/dataz/dsa110/operations/T3/'


def run_filplot(a, wait=False):
    """ Given candidate dictionary, run filterbank analysis, plotting, and classification ("filplot").
    Returns dictionary with updated fields.
    """

    # set up output dict and datestring
    output_dict = a[list(a.keys())[0]]
    output_dict['trigname'] = list(a.keys())[0]
    fill_empty_dict(output_dict)

    # wait for specific filterbank file to be written
    ibeam = output_dict['ibeam'] + 1
    trigname = output_dict['trigname']
    filfile = f"{FILPATH}/{trigname}/{trigname}_{ibeam}.fil"

    print(filfile)
    LOGGER.info('Working on {0}'.format(output_dict['trigname']))
    if wait:
        found_filfile = wait_for_local_file(filfile,TIMEOUT_FIL)
    else:
        found_filfile = filfile if os.path.exists(filfile) else None
    output_dict['filfile'] = found_filfile
    
    if found_filfile is None:
        LOGGER.error('No filfile for {0}'.format(output_dict['trigname']))
        return output_dict
    
    # launch plot and classify
    try:
        output_dict['candplot'], output_dict['probability'], output_dict['real'] = filf.filplot_entry(a, rficlean=False)
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

        return output_dict

    update_dict(output_dict)
    
    return output_dict


def run_burstfit(dd):
    """ Given candidate dictionary, run burstfit analysis.
    Returns new dictionary with refined DM, width, arrival time.
    """

    update_dict(output_dict)

    return dd.copy()


def run_hires(dd):
    """ Given candidate dictionary, create high time/freq filterbanks.
    Returns new dictionary with new file locations?
    """

    update_dict(output_dict)

    return dd.copy()


def run_hdf5copy(d_fp):
    """ Given filplot candidate dictionary, copy hdf5 files
    """

    update_dict(output_dict)
    
    return d_fp.copy()


def run_voltagecopy(d_fp):
    """ Given filplot candidate dictionary, copy voltage files
    """

    update_dict(output_dict)
    
    return d_fp.copy()


def run_hires(d_bf, d_vc):
    """ Given burstfit and voltage dictionaries, generate hires filterbank files.
    """

    dd = d_bf.copy()
    dd.update(d_vc)
    
    update_dict(dd)

    return dd


def run_pol(d_hr):
    """ Given hires candidate dictionary, run polarization analysis.
    Returns new dictionary with new file locations?
    """

    update_dict(d_hr)
    
    return d_hr.copy()


def run_fieldmscopy(d_fp):
    """ Given filplot candidate dictionary, copy field MS file.
    Returns new dictionary with new file locations.
    """

    update_dict(d_fp)

    return d_fp.copy()


def run_candidatems(d_bf, d_vc):
    """ Given filplot and voltage copy candidate dictionaries, make candidate MS image.
    Returns new dictionary with new file locations.
    """

    dd = d_bf.copy()
    dd.update(d_vc)

    update_dict(dd)

    return dd


def run_pol(d_hr):
    """ Given hires candidate dictionary, run polarization analysis.
    Returns new dictionary with new file locations.
    """

    update_dict(d_hr)

    return d_hr.copy()


def run_hiresburstfit(d_hr):
    """ Given hires candidate dictionary, run highres burstfit analysis.
    Returns new dictionary with new file locations.
    """

    update_dict(d_hr)

    return d_hr.copy()


def run_imloc(d_cm):
    """ Given candidate image MS, run image localization.
    """

    update_dict(d_cm)

    return d_cm.copy()


def run_imloc(d_cm):
    """ Given candidate image MS, run image localization.
    """

    update_dict(d_cm)

    return d_cm.copy()


def run_astrometry(d_fm, d_cm):
    """ Given field image MS and candidate image MS, run astrometric localization analysis.
    """

    dd = d_fm.copy()
    dd.update(d_cm)

    update_dict(dd)

    return dd


def run_final(d_h5, d_po, d_hb, d_il):
    """ Token task to handle all final tasks in graph.
    May also update etcd to notify of completion.
    """

    dd = d_h5.copy()
    dd.update(d_po)
    dd.update(d_hb)
    dd.update(d_il)

    update_dict(dd)

    return dd


def update_dict(dd, lock=LOCK):
    """ Read, write, unlock dict on disk with file lock.
    Uses trigname field to find file
    """

    with lock:
        with open(OUTPUT_PATH + dd['trigname'] + '.json', 'w') as f: #encoding='utf-8'                  
            json.dump(output_dict, f, ensure_ascii=False, indent=4)

        
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
        

def wait_for_local_file(fl, timeout):
    """ Wait for file named fl to be written.
    If timeout (in seconds) exceeded, then return None.
    """

    time_counter = 0
    while not os.path.exists(fl):
        time.sleep(1)
        time_counter += 1
        if time_counter > timeout:
            return None

    # wait in case file hasn't been written
    time.sleep(10)

    return fl


