from time import sleep
from dask.distributed import Client
from dsautils import dsa_store
from dsaT3 import T3_manager
import glob, os, json
from dsautils import dsa_functions36

client = Client('10.42.0.232:8786')
de = dsa_store.DsaStore()


tasks = []
def cb_func(dd):
    global tasks
    corrname = dd['corrname']
    trigger = dd['trigger']
    if corrname == 'corr03':
        res = client.submit(task, trigger)
        tasks.append(res)

def datestring_func():
    def a(event):
        global datestring
        datestring = event
    return a

def docopy_func():
    def a(event):
        global docopy
        global candnames
        if event=='True':
            docopy=True
        if event=='False':
            docopy=False
            candnames = []
    return a


# add callbacks from etcd                                                                                
docopy = de.get_dict('/cmd/corr/docopy') == 'True'
datestring = de.get_dict('/cnf/datestring')
de.add_watch('/cnf/datestring', datestring_func())
de.add_watch('/cmd/corr/docopy', docopy_func())

# work through candidates as they are written to disk
candnames = []

while True:
    # get list of triggers in T2, but not in T3
    trig_jsons = sorted(glob.glob(f'/dataz/dsa110/operations/T2/cluster_output/cluster_output*.json'))
    trig_candnames = [fl.split('/')[-1].lstrip('cluster_output').split('.')[0] for fl in trig_jsons]
    t3_jsons = sorted(glob.glob(f'/dataz/dsa110/operations/T3/*.json'))
    t3_candnames = [fl.split('/')[-1].split('.')[0] for fl in t3_jsons]
    trig_jsons = [fl for fl, cn in zip(trig_jsons, trig_candnames) if cn not in t3_candnames]
    print(f"Found {len(trig_jsons)} trigger jsons to process")
    
    for fl in trig_jsons:
        with open(fl) as fp:
            d = json.load(fp)
        candname = list(d.keys())[0]  # format written by initial trigger
#        candname = d['trigname']  # should write it this way for consistency downstream

#        if docopy is True:
        if candname not in candnames:
            print(f"Submitting task for candname {candname}")
            d_fp = client.submit(T3_manager.run_filplot, d, wait=True)  # filplot and classify
            d_bf = client.submit(T3_manager.run_burstfit, d_fp)  # burstfit model fit
            d_vc = client.submit(T3_manager.run_voltagecopy, d_fp)  # copy voltages
            d_h5 = client.submit(T3_manager.run_hdf5copy, d_fp)  # copy hdf5
            d_fm = client.submit(T3_manager.run_fieldmscopy, d_fp)  # copy field image MS
            d_hr = client.submit(T3_manager.run_hires, (d_bf, d_vc))  # create high resolution filterbank
            d_cm = client.submit(T3_manager.run_candidatems, (d_bf, d_vc))  # make candidate image MS
            d_po = client.submit(T3_manager.run_pol, d_hr)  # run pol analysis on hires filterbank
            d_hb = client.submit(T3_manager.run_hiresburstfit, d_hr)  # run burstfit on hires filterbank
            d_il = client.submit(T3_manager.run_imloc, d_cm)  # run image localization on candidate image MS
            d_as = client.submit(T3_manager.run_astrometry, (d_fm, d_cm))  # astrometric burst image
            fut = client.submit(T3_manager.run_final, (d_h5, d_po, d_hb, d_il, d_as))
            tasks.append(fut)
            candnames.append(candname)        

    try:
        print(f'{len(tasks)} tasks in queue for candnames {candnames}')
        if len(tasks)==0:
            candnames = []
        for future in tasks:
            if future.done():
                dd = future.result()
                print(f'\tTask complete for {dd["trigname"]}')
                tasks.remove(future)
                candnames.remove(dd["trigname"])

        de.put_dict('/mon/service/T3manager',{'cadence': 5, 'time': dsa_functions36.current_mjd()})
        sleep(5)
    except KeyboardInterrupt:
        print(f'Cancelling {len(tasks)} tasks and exiting')
        for future in tasks:
            future.cancel()
            tasks.remove(future)
        break
