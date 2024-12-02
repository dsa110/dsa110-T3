from time import sleep
from dsautils import dsa_store
from dsaT3 import T3_manager
import glob, os, json
from dsautils import dsa_functions36
from event import event
from dask.distributed import Client

cl = Client('10.42.0.232:8786')
de = dsa_store.DsaStore()

# work through candidates as they are written to disk
tasks = []

while True:
    # get list of triggers in T2, but not in T3
    trig_jsons = sorted(glob.glob('/dataz/dsa110/operations/T2/cluster_output/cluster_output*.json'))
    trig_candnames = [fl.split('/')[-1].lstrip('cluster_output').split('.')[0] for fl in trig_jsons]
    t3_jsons = sorted(glob.glob('/dataz/dsa110/operations/T3/*.json'))
    t3_candnames = [fl.split('/')[-1].split('.')[0] for fl in t3_jsons]
    trig_jsons = [fl for fl, cn in zip(trig_jsons, trig_candnames) if cn not in t3_candnames]
    candnames = list(set([val.split('-')[1] for grp in cl.processing().values() for val in grp]))
    print(f"Found {len(trig_jsons)} trigger jsons to process. Processing candnames {candnames}.")

    for fl in trig_jsons:
        try:
            d = event.create_event(fl)
        except json.JSONDecodeError:
            print(f'{fl} could not be parsed into an event. skipping...')

        if d.trigname not in candnames:
            fut = T3_manager.submit_cand(fl)
            tasks.append(fut)

    try:
        print(f"{len(tasks)} tasks in queue")
        for future in tasks:
            if future.done():
                if future.status == 'finished':
                    d = future.result()
                    print(f'\tTask complete for {d.trigname}')
                    tasks.remove(future)
                else:
                    print(f'\tTask {future} failed with status {future.status}')

        de.put_dict('/mon/service/T3manager', {'cadence': 5, 'time': dsa_functions36.current_mjd()})
        sleep(5)
    except KeyboardInterrupt:
        print(f'Cancelling {len(tasks)} tasks and exiting')
        for future in tasks:
            future.cancel()
            tasks.remove(future)
        break
