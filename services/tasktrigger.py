from time import sleep
from dask.distributed import Client, Lock
from dsautils import dsa_store
from dsaT3 import T3_manager
import glob, os, json
from dsautils import dsa_functions36

client = Client('10.42.0.232:8786')
de = dsa_store.DsaStore()
LOCK = Lock('update_json')

# work through candidates as they are written to disk
candnames = []
tasks = []

while True:
    # get list of triggers in T2, but not in T3
    trig_jsons = sorted(glob.glob('/dataz/dsa110/operations/T2/cluster_output/cluster_output*.json'))
    trig_candnames = [fl.split('/')[-1].lstrip('cluster_output').split('.')[0] for fl in trig_jsons]
    t3_jsons = sorted(glob.glob('/dataz/dsa110/operations/T3/*.json'))
    t3_candnames = [fl.split('/')[-1].split('.')[0] for fl in t3_jsons]
    trig_jsons = [fl for fl, cn in zip(trig_jsons, trig_candnames) if cn not in t3_candnames]
    print(f"Found {len(trig_jsons)} trigger jsons to process")

    for fl in trig_jsons:
        with open(fl) as fp:
            d = json.load(fp)
        candname = list(d.keys())[0]  # format written by initial trigger
        d = d[candname]
        d['trigname'] = candname

        if candname not in candnames:
            print(f"Submitting task for candname {candname}")
            d_fp = client.submit(T3_manager.run_filplot, d, wait=True, lock=LOCK, resources={'MEMORY': 10e9}, priority=-1)  # filplot and classify
            d_cs = client.submit(T3_manager.run_createstructure, d_fp, lock=LOCK, priority=1)  # burstfit model fit
            d_bf = client.submit(T3_manager.run_burstfit, d_fp, lock=LOCK, priority=1)  # burstfit model fit
            d_vc = client.submit(T3_manager.run_voltagecopy, d_cs, lock=LOCK)  # copy voltages
            d_h5 = client.submit(T3_manager.run_hdf5copy, d_cs, lock=LOCK)  # copy hdf5
            d_fm = client.submit(T3_manager.run_fieldmscopy, d_cs, lock=LOCK)  # copy field image MS
            d_hr = client.submit(T3_manager.run_hires, (d_bf, d_vc), lock=LOCK)  # create high resolution filterbank
            d_cm = client.submit(T3_manager.run_candidatems, (d_bf, d_vc), lock=LOCK)  # make candidate image MS
            d_po = client.submit(T3_manager.run_pol, d_hr, lock=LOCK)  # run pol analysis on hires filterbank
            d_hb = client.submit(T3_manager.run_hiresburstfit, d_hr, lock=LOCK)  # run burstfit on hires filterbank
            d_il = client.submit(T3_manager.run_imloc, d_cm, lock=LOCK)  # run image localization on candidate image MS
            d_as = client.submit(T3_manager.run_astrometry, (d_fm, d_cm), lock=LOCK)  # astrometric burst image
            fut = client.submit(T3_manager.run_final, (d_h5, d_po, d_hb, d_il, d_as), lock=LOCK)
            tasks.append(fut)
            candnames.append(candname)       

    try:
        print(f'{len(tasks)} tasks in queue for candnames {candnames}')
        if len(tasks)==0:
            candnames = []
        for future in tasks:
            if future.done():
                if future.status == 'finished':
                    dd = future.result()
                    print(f'\tTask complete for {dd["trigname"]}')
                    tasks.remove(future)
                    candnames.remove(dd["trigname"])
                else:
                    print(f'\tTask {future} failed with status {future.status}')

        de.put_dict('/mon/service/T3manager',{'cadence': 5, 'time': dsa_functions36.current_mjd()})
        sleep(5)
    except KeyboardInterrupt:
        print(f'Cancelling {len(tasks)} tasks and exiting')
        for future in tasks:
            future.cancel()
            tasks.remove(future)
        break
