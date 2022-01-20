from dsaT3 import filplot_funcs
import json

def test_filplot_entry():
    datestring = '2021_9_9_14_23_23'
    candname = '210909aadd'
    with open(f'/data/dsa110/T2/{datestring}/cluster_output{candname}.json') as f:
        trigger_dict = json.load(f)
    filplot_funcs.filplot_entry(datestring, trigger_dict)
