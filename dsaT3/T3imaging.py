"""Creating and manipulating measurement sets from T3 visibilities.

Author: Dana Simard, dana.simard@astro.caltech.edu
"""
import yaml
import h5py
import numpy as np
from pkg_resources import resource_filename
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as c
from astropy.coordinates import Angle
from antpos.utils import get_itrf
from pyuvdata import UVData
import casatools as cc
from casacore.tables import table
from dsautils import cnf
from dsamfs.io import initialize_uvh5_file, update_uvh5_file
from dsacalib.ms_io import extract_vis_from_ms
from dsacalib.fringestopping import calc_uvw
import dsacalib.constants as ct
from dsacalib.preprocess import remove_outrigger_delays

PARAMFILE = resource_filename('dsaT3', 'data/T3_parameters.yaml')
with open(PARAMFILE) as YAMLF:
    T3PARAMS = yaml.load(YAMLF, Loader=yaml.FullLoader)['T3corr']

MYCONF = cnf.Conf()
CORRPARAMS = MYCONF.get('corr')
MFSPARAMS = MYCONF.get('fringe')
CORR_ORDER = np.arange(1, 17)
ANTENNA_ORDER = [
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    20,
    19,
    18,
    17,
    16,
    15,
    14,
    13,
    100,
    101,
    102,
    116,
    103,
    12,
    11,
    10,
    9,
    8,
    7,
    6,
    5,
    4,
    3,
    2,
    1,
    104,
    105,
    106,
    107,
    108,
    109,
    110,
    111,
    112,
    113,
    114,
    115,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    117,
    44,
    45,
    46,
    47,
    48,
    49,
]

def get_mjd(armed_mjd, utc_start, specnum):
    """Get the start mjd of a voltage dump.

    Parameters
    ----------
    armed_mjd : float
        The time at which the snaps were armed, in mjd.
    utc_start : int
        The spectrum number at which the correlator was started.
    specnum : int
        The spectrum number of the first spectrum in the voltage dump,
        referenced to when the correlator was started.

    Returns
    -------
    tstart : float
        The start time of the voltage dump in mjd.
    """
    tstart = (armed_mjd+utc_start*4*8.192e-6/86400+
              (1/(250e6/8192/2)*specnum/ct.SECONDS_PER_DAY))
    return tstart

def get_blen(antennas):
    """Gets the baseline lengths for a subset of antennas.

    Parameters
    ----------
    antennas : list
        The antennas used in the array.

    Returns
    -------
    blen : array
        The ITRF coordinates of all of the baselines.
    bname : list
        The names of all of the baselines.
    """
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

def generate_T3_uvh5(name, pt_dec, tstart, ntint, nfint, filelist, params=T3PARAMS, start_offset=None, end_offset=None):
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
        The number of frequency bins to bin together before writing the ms
        (compared to the native resolution).
    filelist : dictionary
        The correlator data files for each node.
    params : dictionary
        T3 parameters.
    start_offset : int
        The timesample to start at.  If given, end_offset must also be given.
        Defaults to transform the whole file to a ms.
    end_offset : int
        The timesample to end at.

    Returns
    -------
    str
        The name of the measurement set created.
    """
    antenna_order = params['antennas']
    fobs = params['f0_GHz']+params['deltaf_MHz']*1e-3*(
        np.arange(params['nchan'])+0.5)
    nant = len(antenna_order)
    nbls = (nant*(nant+1))//2
    tsamp = params['deltat_s']*ntint*u.s
    tobs = tstart + (np.arange(params['nsubint']//ntint)+0.5)*tsamp
    if start_offset is None:
        start_offset = 0
    if end_offset is None:
        end_offset = len(tobs)
    #assert start_offset is not None
    #assert end_offset is not None
    tobs = tobs[start_offset:end_offset]
    blen, bname = get_blen(params['antennas'])
    itemspframe = nbls*params['nchan_corr']*params['npol']*2
    framespblock = 16
    itemspblock = itemspframe*framespblock
    assert (end_offset - start_offset)%framespblock == 0
    nblocks = (end_offset-start_offset)//framespblock
    # Get outrigger delays
    delays = np.zeros(len(bname), dtype=np.int)
    for i, bn in enumerate(bname):
        ant1, ant2 = bn.split('-')
        delays[i] = MFSPARAMS['outrigger_delays'].get(int(ant1), 0)-\
                    MFSPARAMS['outrigger_delays'].get(int(ant2), 0)
    for corr, corrfile in filelist.items(): # corr, ch0 in params['ch0'].items():
        ch0 = params['ch0'][corr]
        fobs_corr_full = fobs[ch0:(ch0+params['nchan_corr'])]
        fobs_corr = np.median(fobs_corr_full.reshape(-1, nfint), axis=-1)
        outname = '{1}_{0}.hdf5'.format(corr, name)
        with h5py.File(outname, 'w') as fhdf5:
            initialize_uvh5_file(
                fhdf5,
                len(fobs_corr),
                2,
                pt_dec.to_value(u.rad),
                antenna_order,
                fobs_corr,
                #outrigger_delays
            )
            with open(corrfile, 'rb') as cfhandler:
                if start_offset is not None:
                    cfhandler.seek(start_offset*32*itemspframe)
                for i in range(nblocks):
                    bu, bv, bw = calc_uvw(
                        blen,
                        tobs.mjd[i*framespblock:(i+1)*framespblock],
                        'HADEC',
                        np.zeros(framespblock)*u.rad,
                        np.ones(framespblock)*pt_dec
                    )
                    buvw = np.array([bu, bv, bw]).T
                    data = np.fromfile(
                        cfhandler,
                        dtype=np.float32,
                        count=itemspblock
                    )
                    data = data.reshape(-1, 2)
                    data = data[..., 0] + 1.j*data[..., 1]
                    data = data.reshape(framespblock, nbls, len(fobs_corr_full), params['npol'])[..., [0, -1]]
                    total_delay = (delays[np.newaxis, :]+(buvw[..., 2]*u.m/c.c).to_value(u.nanosecond))[:, :, np.newaxis, np.newaxis]
                    vis_model = np.exp(2j*np.pi*fobs_corr_full[:, np.newaxis]*total_delay)
                    vis_model = vis_model.astype(np.complex64)
                    data /= vis_model
                    if nfint > 1:
                        data = data.reshape(framespblock, nbls, len(fobs_corr), nfint, 2).mean(axis=3)
                    update_uvh5_file(
                        fhdf5,
                        data.astype(np.complex64),
                        tobs.jd[i*framespblock:(i+1)*framespblock],
                        tsamp,
                        bname,
                        buvw,
                        np.ones(data.shape, np.float32)
                    )
    return outname

def plot_image(imname, verbose=False, outname=None, show=True,
              expected_point=None):
    """Plots an image from the casa-generated image file.

    Paramters
    ---------
    imname : str
        The name full path of the image file.
    verbose : bool
        If set to True, prints some information about the image.
    outname : str
        If provided, saves the image in <outname>_image.png.
    show : bool
        If False, the image is closed at the end of the function.
    cellsize : str
        The size of each pixel, in a Casa-recognized angle.
    """
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
    max_idxs = np.unravel_index(imvals.argmax(), imvals.shape)
    cellsizex = Angle(dd['incr'][0], dd['axisunits'][0])
    cellsizey = Angle(dd['incr'][1], dd['axisunits'][1])
    ra, dec = (
        Angle('{0}{1}'.format(dd['refval'][0], dd['axisunits'][0])),
        Angle('{0}{1}'.format(dd['refval'][1], dd['axisunits'][1]))
    )
    brightest_point = (
        ra +
        Angle('{0}{1}'.format(
            dd['incr'][0]*(max_idxs[0]-dd['refpix'][0]),
            dd['axisunits'][0]
        ))/np.cos(dec),
        dec +
        Angle('{0}{1}'.format(
            dd['incr'][1]*(max_idxs[1]-dd['refpix'][1]),
            dd['axisunits'][1]
        ))
    )
    if verbose:
        print('Peak SNR at pix ({0},{1}) = {2}'.format(max_idxs[0],
                                                       max_idxs[1],
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
            (-imvals.shape[0]/2*Angle(cellsizex)).to_value(u.arcmin),
            (imvals.shape[0]/2*Angle(cellsizex)).to_value(u.arcmin),
            (-imvals.shape[1]/2*Angle(cellsizey)).to_value(u.arcmin),
            (imvals.shape[1]/2*Angle(cellsizey)).to_value(u.arcmin)
        ]
    )
    plt.colorbar(pim)
    ax.axvline(0, color='white', alpha=0.5)
    ax.axhline(0, color='white', alpha=0.5)
    ax.set_xlabel('l (arcmin)')
    ax.set_ylabel('m (arcmin)')
    plttitle = '{0} {1:.2f} {2:.2f}'.format(
        imname,
        brightest_point[0],
        brightest_point[1]
    )
    if expected_point is not None:
        plttitle += ', offset by {0:.2f} {1:.2f}'.format(
            (brightest_point[0]-expected_point[0]).to(u.arcmin),
            (brightest_point[1]-expected_point[1]).to(u.arcmin)
        )
    plt.title(plttitle)
    if outname is not None:
        plt.savefig('{0}_image.png'.format(outname))
    if not show:
        plt.close()
    if error > 0:
        print('{0} errors occured during imaging'.format(error))
    return brightest_point

def read_bfweights(bfweights, bfdir):
    """Reads the beamforming weights.

    Parameters
    ----------
    bfweights : str
        The label of the file containing the weights. Will open
        <bfdir>/beamformer_weights_<bfweights>.yaml
    bfdir : str
        The directory in which the beamformer weights are stored.

    Returns
    -------
    antenna_order : list
        The order of the antennas in the bfweights array.
    bfweights : ndarray
        The beamformer weights, (antenna, freqeuncy, polarization).
        Frequency is in the same order as in the correlator.
    """
    with open('{0}/beamformer_weights_{1}.yaml'.format(
            bfdir,
            bfweights,
    )) as yamlf:
        bfparams = yaml.load(yamlf, Loader=yaml.FullLoader)
    if 'cal_solutions' in bfparams.keys():
        bfparams = bfparams['cal_solutions']
    antenna_order = bfparams.get('antenna_order', ANTENNA_ORDER)
    corr_order = bfparams.get('corr_order', CORR_ORDER)
    gains = np.zeros(
        (len(antenna_order), len(corr_order), 48, 2),
        dtype=np.complex
    )
    for corridx, corr in enumerate(corr_order):
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
        (len(antenna_order), len(corr_order)*48, 2)
    )
    return antenna_order, gains

def calibrate_T3ms(msname, bfweights, bfdir, dedisp_mask=None):
    """Calibrates a measurement set using the beamformer weights.

    Calibrated data is written into the CORRECTED_DATA column.

    Parameters
    ----------
    msname : str
        The name of the measurement set.
    bfweights : str
        The label of the file containing the weights. Will open
        <bfdir>/beamformer_weights_<bfweights>.yaml
    bfdir : str
        The directory in which the beamformer weights are stored.
    dedisp_mask : str
        The path to a dedispersion mask to be applied.
    """
    antenna_order, gains = read_bfweights(bfweights, bfdir)
    gains = gains[:, ::-1, :]

    data, _, fobs, flags, ant1, ant2, _, _, orig_shape = extract_vis_from_ms(
        msname,
        data='data'
    )
    print(data.shape)
    data = data.reshape(
        data.shape[0],
        data.shape[1],
        data.shape[2],
        gains.shape[1],
        -1,
        data.shape[-1]
    )
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
            bl_gains = np.exp(1.j*np.angle(bl_gains))
            data[i, ...] *= bl_gains[:, np.newaxis, :]
        except ValueError:
            flags[i, ...] = 1
            print('no calibration solutions for baseline {0}-{1}'.format(a1, a2))
    data = data.swapaxes(0, 1).reshape((-1, len(fobs), data.shape[-1]))
    flags = flags.swapaxes(0, 1).reshape((-1, len(fobs), flags.shape[-1]))
    # dedisp_flags = np.load(dedisp_mask)
    # check size 
    # data[data!=data] = np.nanmean(data) this should be okay now
    with table('{0}.ms'.format(msname), readonly=False) as tb:
        tb.putcol('CORRECTED_DATA', data)
        tb.putcol('FLAG', flags)
