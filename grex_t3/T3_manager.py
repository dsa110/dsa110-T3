import traceback
import numpy as np
#from dsautils import dsa_store
#import dsautils.dsa_syslog as dsl
from grex_t3 import filplot_funcs as filf
from grex_t3 import data_manager
import time, os
import json
from dask.distributed import Client

import logging as LOGGER
logger.basicConfig(filename='logs/output.log',
                    encoding='utf-8',
                    level=logger.DEBUG)

client = Client('12.0.0.1:8786')
#ds = dsa_store.DsaStore()

TIMEOUT_FIL = 60
FILPATH = '/home/liam/grexdata/'
OUTPUT_PATH = '/home/liam/grexdata/output'


def run_filplot(a, wait=False, lock=None):
    """ Given candidate dictionary, 
    run filterbank analysis, 
    plotting, and classification ("filplot").
    Returns dictionary with updated fields.
    """

    # set up output dict and datestring
    output_dict = a[list(a.keys())[0]]
    output_dict['trigname'] = list(a.keys())[0]
    fill_empty_dict(output_dict)

    print('run_filplot on {0}'.format(output_dict['trigname']))
    LOGGER.info('run_filplot on {0}'.format(output_dict['trigname']))

    # wait for specific filterbank file to be written
    ibeam = output_dict['ibeam'] + 1
    trigname = output_dict['trigname']
    filfile = f"{FILPATH}/{trigname}/{trigname}_{ibeam}.fil"

    # TODO: should be obsolete. remove this and retest. 
    if wait:
        found_filfile = wait_for_local_file(filfile, TIMEOUT_FIL)
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

    update_json(output_dict, lock=lock)
    
    return output_dict


def run_burstfit(dd, lock=None):
    """ Given candidate dictionary, run burstfit analysis.
    Returns new dictionary with refined DM, width, arrival time.
    """

    print('run_burstfit on {0}'.format(dd['trigname']))
    LOGGER.info('run_burstfit on {0}'.format(dd['trigname']))

    update_json(dd, lock=lock)

    return dd.copy()


def run_hdf5copy(d_fp, lock=None):
    """ Given filplot candidate dictionary, copy hdf5 files
    """

    print('run_hdf5copy on {0}'.format(d_fp['trigname']))
    LOGGER.info('run_hdf5copy on {0}'.format(d_fp['trigname']))

    update_json(d_fp, lock=lock)
    
    return d_fp.copy()


def run_voltagecopy(d_fp, lock=None):
    """ Given filplot candidate dictionary, copy voltage files
    """

    print('run_voltagecopy on {0}'.format(d_fp['trigname']))
    LOGGER.info('run_voltagecopy on {0}'.format(d_fp['trigname']))

    update_json(d_fp, lock=lock)
    
    return d_fp.copy()


def run_hires(dds, lock=None):
    """ Given burstfit and voltage dictionaries, generate hires filterbank files.
    """

    d_bf, d_vc = dds
    dd = d_bf.copy()

    print('run_hires on {0}'.format(dd['trigname']))
    LOGGER.info('run_hires on {0}'.format(dd['trigname']))

    dd.update(d_vc)
    
    update_json(dd, lock=lock)

    return dd


def run_pol(d_hr, lock=None):
    """ Given hires candidate dictionary, run polarization analysis.
    Returns new dictionary with new file locations?
    """

    print('run_pol on {0}'.format(d_hr['trigname']))
    LOGGER.info('run_pol on {0}'.format(d_hr['trigname']))

    update_json(d_hr, lock=lock)
    
    return d_hr.copy()


def run_fieldmscopy(d_fp, lock=None):
    """ Given filplot candidate dictionary, copy field MS file.
    Returns new dictionary with new file locations.
    """

    print('run_fieldmscopy on {0}'.format(d_fp['trigname']))
    LOGGER.info('run_fieldmscopy on {0}'.format(d_fp['trigname']))

    update_json(d_fp, lock=lock)

    return d_fp.copy()


def run_candidatems(dds, lock=None):
    """ Given filplot and voltage copy candidate dictionaries, make candidate MS image.
    Returns new dictionary with new file locations.
    """

    d_bf, d_vc = dds
    dd = d_bf.copy()

    print('run_candidatems on {0}'.format(dd['trigname']))
    LOGGER.info('run_candidatems on {0}'.format(dd['trigname']))

    dd.update(d_vc)

    update_json(dd, lock=lock)

    return dd


def run_hiresburstfit(d_hr, lock=None):
    """ Given hires candidate dictionary, run highres burstfit analysis.
    Returns new dictionary with new file locations.
    """

    print('run_hiresburstfit on {0}'.format(d_hr['trigname']))
    LOGGER.info('run_hiresburstfit on {0}'.format(d_hr['trigname']))

    update_json(d_hr, lock=lock)

    return d_hr.copy()


def run_imloc(d_cm, lock=None):
    """ Given candidate image MS, run image localization.
    """

    print('run_imloc on {0}'.format(d_cm['trigname']))
    LOGGER.info('run_imloc on {0}'.format(d_cm['trigname']))

    update_json(d_cm, lock=lock)

    return d_cm.copy()


def run_astrometry(dds, lock=None):
    """ Given field image MS and candidate image MS, run astrometric localization analysis.
    """

    d_fm, d_cm = dds
    dd = d_fm.copy()

    print('run_astrometry on {0}'.format(dd['trigname']))
    LOGGER.info('run_astrometry on {0}'.format(dd['trigname']))

    dd.update(d_cm)

    update_json(dd, lock=lock)

    return dd


def run_final(dds, lock=None):
    """ Token task to handle all final tasks in graph.
    May also update etcd to notify of completion.
    """

    d_h5, d_po, d_hb, d_il, d_as = dds
    dd = d_h5.copy()

    print('run_final on {0}'.format(dd['trigname']))
    LOGGER.info('run_final on {0}'.format(dd['trigname']))

    dd.update(d_po)
    dd.update(d_hb)
    dd.update(d_il)

    # do data management
    dm = data_manager.DataManager(dd)
    dd = dm()

    update_json(dd, lock=lock)
    return dd


def update_json(dd, lock, outpath=OUTPUT_PATH):
    """ Lock, read, write, unlock json file on disk.
    Uses trigname field to find file
    """

    fn = outpath + dd['trigname'] + '.json'

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

    lock.release()


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


