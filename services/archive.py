from dsaT3 import utils
from dsautils import dsa_store
ds = dsa_store.DsaStore()

datestring = ds.get_dict('/cnf/datestring')
utils.archive(datestring)
