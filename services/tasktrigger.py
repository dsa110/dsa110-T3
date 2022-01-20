from time import sleep
from dask.distributed import Client
from dsautils import dsa_store
from dsaT3 import T3_manager
import glob, os, json
from dsautils import dsa_functions36

client = Client('10.42.0.232:8786')
de = dsa_store.DsaStore()

def task(a):

    T3dict = T3_manager.run(a)
    return T3dict

def task_nowait(a):

    T3dict = T3_manager.run_nowait(a)
    return T3dict


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

    trig_jsons = sorted(glob.glob('/data/dsa110/T2/'+datestring+'/cluster_output*.json'))
    for fl in trig_jsons:
        f = open(fl)
        d = json.load(f)
        trigname = list(d.keys())[0]

        if docopy is True:
            if trigname not in candnames:
                if len(tasks)<8:
                    candnames.append(trigname)        
                    if not os.path.exists('/home/ubuntu/data/T3/'+trigname+'.png'):
                        res = client.submit(task_nowait, d)
                        tasks.append(res)
    
    try:
        print(f'{len(tasks)} tasks in queue')
        if len(tasks)==0:
            candnames = []
        for future in tasks:
            if future.done():
                dd = future.result()
                print(f'\tTask complete for {dd["trigname"]}')
                tasks.remove(future)

        de.put_dict('/mon/service/T3manager',{'cadence': 5, 'time': dsa_functions36.current_mjd()})
        sleep(5)
    except KeyboardInterrupt:
        print(f'Cancelling {len(tasks)} tasks and exiting')
        for future in tasks:
            future.cancel()
            tasks.remove(future)
        break
