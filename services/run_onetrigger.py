from time import sleep
from dsautils import dsa_store
from dsaT3 import T3_manager
import glob, os, json
from dsautils import dsa_functions36
import sys

de = dsa_store.DsaStore()
datestring = de.get_dict('/cnf/datestring')

trig_jsons = '/data/dsa110/T2/'+datestring+'/cluster_output'+sys.argv[1]+'.json'
f = open(trig_jsons)
d = json.load(f)
trigname = list(d.keys())[0]
print(trigname)
T3_manager.run_filplot(d)

    
    
