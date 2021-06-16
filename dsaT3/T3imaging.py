"""Creating and manipulating measurement sets from T3 visibilities.
"""
import yaml
import h5py
import numpy as np
from pkg_resources import resource_filename
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import Angle
from antpos.utils import get_itrf
from pyuvdata import UVData
import casatools as cc
from casacore.tables import table
from dsautils import cnf
from dsamfs.io import initialize_uvh5_file, update_uvh5_file
from dsacalib.ms_io import uvh5_to_ms, extract_vis_from_ms
from dsacalib.fringestopping import calc_uvw
import dsacalib.constants as ct
from dsacalib.preprocess import remove_outrigger_delays

PARAMFILE = resource_filename('dsaT3', 'data/T3_parameters.yaml')
with open(PARAMFILE) as YAMLF:
    T3PARAMS = yaml.load(YAMLF, Loader=yaml.FullLoader)['T3corr']

MYCONF = cnf.Conf()
CORRPARAMS = MYCONF.get('corr')

def get_mjd(armed_mjd, utc_start, specnum):
    tstart = (armed_mjd+utc_start*4*8.192e-6/86400+
              (1/(250e6/8192/2)*specnum/ct.SECONDS_PER_DAY))
    return tstart

def get_blen(antennas):
    ant_itrf = get_itrf(
        latlon_center=(ct.OVRO_LAT*u.rad, ct.OVRO_LON*u.rad, ct.OVRO_ALT*u.m)
    ).loc[antennas]
    xx = np.array(ant_itrf['dx_m'])
    yy = np.array(ant_itrf['dy_m'])
    zz = np.array(ant_itrf['dz_m'])
    # Get uvw coordinates
    nants = len(antennas)
    nbls = (nants*(nants+1))//2
    blen = np.zeros((nbls, 3))
    bname = []
    k = 0
    for i in range(nants):
        for j in range(i, nants):
            blen[k, :] = np.array([
                xx[i]-xx[j],
                yy[i]-yy[j],
                zz[i]-zz[j]
            ])
            bname += ['{0}-{1}'.format(
                antennas[i],
                antennas[j]
            )]
            k += 1
    return blen, bname

def generate_T3_ms(name, pt_dec, tstart, ntint, nfint, filelist, params=T3PARAMS, start_offset=None, end_offset=None):
    """Generates a measurement set from the T3 correlations.

    Parameters
    ----------
    name : str
        The name of the measurement set.
    pt_dec : quantity
        The pointing declination in degrees or equivalient.
    tstart : astropy.time.Time instance
        The start time of the correlated data.
    ntint : float
        The number of time bins that have been binned together (compared to the
        native correlator resolution).
    nfint : float
        The number of frequency bins that have been binned together (compared
        to the native resolution).
    filelist : dictionary
        The correlator data files for each node.
    params : dictionary
        T3 parameters.
    """
    msname = '{0}/{1}'.format(params['msdir'], name)
    antenna_order = params['antennas']
    fobs = params['f0_GHz']+params['deltaf_MHz']*1e-3*nfint*(
        np.arange(params['nchan']//nfint)+0.5)
    antenna_order = params['antennas']
    nant = len(antenna_order)
    nbls = (nant*(nant+1))//2
    tsamp = params['deltat_s']*ntint*u.s
    tobs = tstart + (np.arange(params['nsubint']//ntint)+0.5)*tsamp
    if start_offset is not None:
        assert end_offset is not None
        tobs = tobs[start_offset:end_offset]
    blen, bname = get_blen(params['antennas'])
    bu, bv, bw = calc_uvw(
        blen,
        tobs.mjd,
        'HADEC',
        np.zeros(len(tobs))*u.rad,
        np.ones(len(tobs))*pt_dec
    )
    buvw = np.array([bu, bv, bw]).T
    hdf5_files = []
    for corr, ch0 in params['ch0'].items():
        fobs_corr = fobs[ch0//nfint:(ch0+params['nchan_corr'])//nfint]
        data = np.fromfile(
            filelist[corr],
            dtype=np.float32
        )
        data = data.reshape(-1, 2)
        data = data[..., 0] + 1.j*data[..., 1]
        data = data.reshape(-1, nbls, len(fobs_corr), 4)[..., [0, -1]]
        if start_offset is not None:
            data = data[start_offset:end_offset, ...]
        outname = '{2}/{1}_{0}.hdf5'.format(corr, name, params['msdir'])
        with h5py.File(outname, 'w') as fhdf5:
            initialize_uvh5_file(
                fhdf5,
                len(fobs_corr),
                2,
                pt_dec.to_value(u.rad),
                antenna_order,
                fobs_corr
            )
            update_uvh5_file(
                fhdf5,
                data,
                tobs.jd,
                tsamp,
                bname,
                buvw,
                np.ones(data.shape, np.float32)
            )
        UV = UVData()
        UV.read(outname, file_type='uvh5')
        remove_outrigger_delays(UV)
        UV.write_uvh5(outname, clobber=True)
        hdf5_files += [outname]
    uvh5_to_ms(
        hdf5_files,
        msname
    )
    return msname

def plot_image(imname, verbose=False, outname=None, show=True, cellsize='0.2arcsec'):
    """Plots an image from the casa-generated image file.
    """
    # TODO: Get cell-size from the image data
    error = 0
    ia = cc.image()
    error += not ia.open(imname)
    dd = ia.summary()
    # dd has shape npixx, npixy, nch, npol
    npixx = dd['shape'][0]
    if verbose:
        print('Image shape: {0}'.format(dd['shape']))
    imvals = ia.getchunk(0, int(npixx))[:, :, 0, 0]
    #imvals = fftshift(imvals)
    error += ia.done()
    if verbose:
        peakx, peaky = np.where(imvals.max() == imvals)
        print('Peak SNR at pix ({0},{1}) = {2}'.format(peakx[0], peaky[0],
                                                       imvals.max()/
                                                       imvals.std()))
        print('Value at peak: {0}'.format(imvals.max()))
        print('Value at origin: {0}'.format(imvals[imvals.shape[0]//2,
                                                   imvals.shape[1]//2]))

    _, ax = plt.subplots(1, 1, figsize=(15, 8))
    pim = ax.imshow(
        imvals.transpose(),
        interpolation='none',
        origin='lower',
        extent=[
            (-imvals.shape[0]/2*Angle(cellsize)).to_value(u.arcsecond),
            (imvals.shape[0]/2*Angle(cellsize)).to_value(u.arcsecond),
            (-imvals.shape[1]/2*Angle(cellsize)).to_value(u.arcsecond),
            (imvals.shape[1]/2*Angle(cellsize)).to_value(u.arcsecond)
        ]
    )
    plt.colorbar(pim)
    ax.axvline(0, color='white', alpha=0.5)
    ax.axhline(0, color='white', alpha=0.5)
    ax.set_xlabel('l (arcsec)')
    ax.set_ylabel('m (arcsec)')
    if outname is not None:
        plt.savefig('{0}_image.png'.format(outname))
    if not show:
        plt.close()
    if error > 0:
        print('{0} errors occured during imaging'.format(error))

def read_bfweights(bfweights, bfdir):
    with open('{0}/beamformer_weights_{1}.yaml'.format(
            bfdir,
            bfweights,
    )) as yamlf:
        bfparams = yaml.load(yamlf, Loader=yaml.FullLoader)
    if 'cal_solutions' in bfparams.keys():
        bfparams = bfparams['cal_solutions']
    gains = np.zeros(
        (len(bfparams['antenna_order']), len(bfparams['corr_order']), 48, 2),
        dtype=np.complex
    )
    for corridx, corr in enumerate(bfparams['corr_order']):
        with open(
                '{0}/beamformer_weights_corr{1:02d}_{2}.dat'.format(
                    bfdir,
                    corr,
                    bfweights
                ),
                'rb'
        ) as f:
            data = np.fromfile(f, '<f4')
        temp = data[64:].reshape(64, 48, 2, 2)
        gains[:, corridx, :, :] = temp[..., 0]+1.0j*temp[..., 1]
    gains = gains.reshape(
        (len(bfparams['antenna_order']), len(bfparams['corr_order'])*48, 2)
    )
    return bfparams['antenna_order'], gains

def calibrate_T3ms(msname, bfweights, bfdir):
    antenna_order, gains = read_bfweights(bfweights, bfdir)
    gains = gains[:, ::-1, :]

    data, _, fobs, flags, ant1, ant2, _, _, orig_shape = extract_vis_from_ms(
        msname,
        data='data'
    )
    data = data.reshape(data.shape[0], data.shape[1], data.shape[2], gains.shape[1], -1, data.shape[-1])
    assert np.all(np.diff(fobs) > 0)
    assert orig_shape == ['time', 'baseline', 'spw']
    for i in range(data.shape[0]):
        a1 = ant1[i]+1
        a2 = ant2[i]+1
        try:
            bl_gains = (
                np.conjugate(
                    gains[antenna_order.index(a2), ...]
                )*gains[antenna_order.index(a1), ...]
            )
            bl_gains = bl_gains/np.abs(bl_gains)
            data[i, ...] *= bl_gains[:, np.newaxis, :]
        except ValueError:
            flags[i, ...] = 1
            print('no calibration solutions for baseline {0}-{1}'.format(a1, a2))
    data = data.swapaxes(0, 1).reshape((-1, len(fobs), data.shape[-1]))
    flags = flags.swapaxes(0, 1).reshape((-1, len(fobs), flags.shape[-1]))

    with table('{0}.ms'.format(msname), readonly=False) as tb:
        tb.putcol('CORRECTED_DATA', data)
        tb.putcol('FLAG', flags)
