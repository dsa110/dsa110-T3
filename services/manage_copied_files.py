from time import sleep
from dsautils import dsa_store
import glob, os, json
from dsautils import dsa_functions36

de = dsa_store.DsaStore()
datestring = de.get_dict('/cnf/datestring')
odir = "/media/ubuntu/ssd/T3/"+datestring
#odir = "/media/ubuntu/ssd/T3"
MEMFL = "/home/ubuntu/data/T3/management.json"
os.system("rm -rf "+MEMFL)
TRIGGER_WAIT = 7200./86400.
FSIZE = 2972712960
MIN_CT = 8

# a is file name
def extract_cand_from_json(a):

    f = open(a)
    j = json.load(f)
    f.close()
    return list(j.keys())[0]

def add_empty_dict(MEM,cname):

    # find dict
    hfls = glob.glob(odir+"/corr*/"+cname+"_header.json")
    hfls.sort(key=os.path.getmtime, reverse=True)
    hfl = open(hfls[0])
    di = json.load(hfl)
    
    for corr in ['corr03','corr04','corr05','corr06','corr07','corr08','corr10','corr11','corr12','corr14','corr15','corr16','corr18','corr19','corr21','corr22']:
        di[cname][corr+'_data'] = None
        di[cname][corr+'_header'] = None
    
    MEM[cname] = di
    MEM[cname+"_trigger"] = False

def update_mem(MEM,cname):

    ct = 0
    for corr in ['corr03','corr04','corr05','corr06','corr07','corr08','corr10','corr11','corr12','corr14','corr15','corr16','corr18','corr19','corr21','corr22']:

        fl = odir+"/"+corr+"/"+cname+"_data.out"
        if os.path.exists(fl):
            if os.path.getsize(fl)==FSIZE:
                MEM[cname][cname][corr+'_data'] = fl
                ct += 1
        fl = odir+"/"+corr+"/"+cname+"_header.json"
        if os.path.exists(fl):
            MEM[cname][cname][corr+'_header'] = fl

    MEM[cname+"_ct"] = ct
    

while True:

    # load dict from disk if present
    if os.path.exists(MEMFL):
        f = open(MEMFL)
        MEM = json.load(f)
    else:
        MEM = {}
    
    # get list of files sorted by write time
    hdrs = glob.glob(odir+"/corr*/*_header.json")
    hdrs.sort(key=os.path.getmtime, reverse=True)
    #volts = glob.glob(odir+"/*_data.out")
    #volts.sort(key=os.path.getmtime, reverse=True)
    
    # using only headers, identify unique candidate names
    candnames = []
    for hdr in hdrs:
        cname = extract_cand_from_json(hdr)
        if cname not in candnames:
            candnames.append(cname)

    # for each candidate search for associated voltage files
    # update dict
    for cname in candnames:

        # add to dict
        if cname not in list(MEM.keys()):
            add_empty_dict(MEM,cname)
            MEM[cname+'_mjd'] = dsa_functions36.current_mjd()

        # search for associated voltage files
        if MEM[cname+'_trigger'] is False:
            update_mem(MEM,cname)


    # triggering!
    for cname in candnames:

        # send trigger onwards if all 16 voltage files present
        if MEM[cname+'_ct']==16:
            if MEM[cname+'_trigger'] is False:
                de.put_dict('/mon/corr/1/voltagecopy',MEM[cname])
                MEM[cname+'_trigger'] = True
                MEM[cname+'_trigger_time'] = dsa_functions36.current_mjd()
                
        # send trigger onwards if 30 min have passed since entry was created
        if dsa_functions36.current_mjd()-MEM[cname+'_mjd'] > TRIGGER_WAIT:
            if MEM[cname+'_ct']>=MIN_CT:
                if MEM[cname+'_trigger'] is False:
                    de.put_dict('/mon/corr/1/voltagecopy',MEM[cname])
                    MEM[cname+'_trigger'] = True
                    MEM[cname+'_trigger_time'] = dsa_functions36.current_mjd()
            

    # write MEM to disk and sleep
    with open(MEMFL, 'w') as f:
        json.dump(MEM, f)
    f.close()

    
    sleep(10)
