import sys
from astropy.time import Time
from dsautils.coordinates import get_pointing, get_galcoord
from catalogs.psrtools import QUERY, match_pulsar

if __name__=='__main__':
    if len(sys.argv)==1:
        ra,dec,mjd = get_pointing()
    elif len(sys.argv)==2:
        mjd = float(sys.argv[1])
        ra,dec = get_pointing(Time(mjd, format='mjd'))

    get_galcoord(ra.value, dec.value)
    
    ind_near = match_pulsar(ra, dec, thresh_deg=3.5)

    print("\n\nMJD: %0.5f"%mjd)
    print("RA and Dec: %0.2f %0.2f" % (ra.value,dec.value))
    if len(ind_near)==0:
        print("There are no pulsars within 3.5deg of beam center")
    else:
        print("There is/are %d pulsar(s) within 3.5deg of beam center:"%len(ind_near))

    for ii in ind_near:
        name_psrb = QUERY['PSRB'][ii]
        name_psrj = QUERY['PSRJ'][ii]
        dm = QUERY['DM'][ii]
        flux = QUERY['S1400'][ii]
        print('    %s/%s with DM=%0.1f pc cm**-3 and S1400=%0.3fmJy'%(name_psrb, name_psrj, dm, flux))
    

