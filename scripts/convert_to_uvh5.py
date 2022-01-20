from dsacalib.ms_io import uvh5_to_ms
import glob

for name in ['hydraAaaa_full']:
    msname = '/media/ubuntu/data/dsa110/imaging/{0}'.format(name)
    uvh5files = sorted(glob.glob(
        '/media/ubuntu/data/dsa110/imaging/{0}_corr??.hdf5'.format(name)
    ))
    print(msname)
    print(len(uvh5files))
    print(uvh5files)
    uvh5_to_ms(
        uvh5files,
        msname
    )
