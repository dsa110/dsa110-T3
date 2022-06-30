"""Data manager and related functions for managing Level 1 data in T3."""

from types import MappingProxyType
from pathlib import Path
from copy import deepcopy
from itertools import chain
import re
from astropy.time import Time
import astropy.units as u
from dsautils import cnf
from dsautils import dsa_syslog as dsl


class DataManager:
    """Manage Level 1 data for confirmed candidates."""

    operations_dir = Path("/dataz/dsa110/operations")
    candidates_dir = Path("/dataz/dsa110/candidates")
    candidates_subdirs = (
        "Level3", "Level2/voltages", "Level2/filterbank",
        "Level2/calibration", "other")
    try:
        subband_corrnames = tuple(cnf.Conf().get('corr')['ch0'].keys())
    except:
        subband_corrnames = None
    nbeams = 256

    # Ensure read only since shared between instances
    directory_structure = MappingProxyType({
        'voltages':
            {
                'target': (
                    "{operations_dir}/T3/voltages/{hostname}_{candname}_"
                    "data.out"),
                'destination': (
                    "{candidates_dir}/{candname}/Level2/voltages/{candname}_"
                    "{subband}_data.out"),
            },
        'filterbank':
            {
                'target': (
                    "{operations_dir}/T1/{candname}/{candname}_"
                    "{beamnumber}.fil"),
                'destination': (
                    "{candidates_dir}/{candname}/Level2/filterbank/{candname}_"
                    "{beamnumber}.fil"),
            },
        'beamformer_weights':
            {
                'target': "{operations_dir}/beamformer_weights/applied/",
                'destination': (
                    "{candidates_dir}/{candname}/Level2/calibration/")
            },
        'hdf5_files':
            {
                'target': "{operations_dir}/correlator/{hdf5_name}*.hdf5",
                'destination': "{candidates_dir}/{candname}/Level3/"
            },
        'T2_csv':
            {
                'target': "{operations_dir}/T2/cluster_output.csv",
                'destination': (
                    "{candidates_dir}/{candname}/Level2/T2_{candname}.csv")
            },
        'filplot_json':
            {
                'target': "{operations_dir}/T3/{candname}.json",
                'destination': "{candidates_dir}/{candname}/Level3/{candname}.json"
            },
        'filplot_png':
            {
                'target': "{operations_dir}/T3/{candname}.png",
                'destination': "{candidates_dir}/{candname}/other/{candname}.png"
            }
    })

    def __init__(
            self, candparams: dict, logger: dsl.DsaSyslogger = None) -> None:
        """Initialize info from candidate.

        Parameters
        ----------
        candparams : dict
            Dictionary of candidate parameters.
        logger : dsl.DsaSyslogger
            Logger object.
        """
        if logger is None:
            self.logger = dsl.DsaSyslogger()
        else:
            self.logger = logger

        self.candname = candparams['trigname']
        self.candparams = deepcopy(candparams)
        self.candtime = Time(
            self.candparams['mjds'], format='mjd', precision=0)

    def __call__(self) -> dict:
        """Create the candidate directory structure and hardlink pre-existing files.

        Returns
        -------
        dict
            Dictionary of candidate parameters.
        """
        self.create_directory_structure()
        self.link_filterbank()
        self.link_beamformer_weights()
        self.link_T2_csv()
        self.link_filplot_and_json()

        return self.candparams

    def create_directory_structure(self) -> None:
        """Create directory structure for candidate."""

        self.logger.info(
            f"Creating directory structure for candidate {self.candname}.")

        cand_dir = self.candidates_dir / self.candname
        for subdir in self.candidates_subdirs:
            newdir = cand_dir / subdir
            if not newdir.exists():
                newdir.mkdir(parents=True)

        self.logger.info(
            f"Directory structure at {cand_dir} created for {self.candname}.")

    def link_voltages(self) -> None:
        """Link voltages to candidate directory."""

        self.logger.info(
            f"Linking voltages to candidate directory for {self.candname}.")

        for subband, corrname in enumerate(self.subband_corrnames):
            sourcepath = Path(
                self.directory_structure['voltages']['target'].format(
                    operations_dir=self.operations_dir, candname=self.candname,
                    hostname=corrname))
            destpath = Path(
                self.directory_structure['voltages']['destination'].format(
                    candidates_dir=self.candidates_dir, candname=self.candname,
                    subband=f"sb{subband:02d}"))
            self.link_file(sourcepath, destpath)
            self.candparams[f'voltage_sb{subband:02d}'] = str(destpath)

        self.logger.info(f"Voltages linked for {self.candname}.")

    def link_filterbank(self) -> None:
        """Link filterbank to candidate directory."""

        self.logger.info(
            f"Linking filterbank to candidate directory for {self.candname}.")

        for beamnumber in range(self.nbeams):
            sourcepath = Path(
                self.directory_structure['filterbank']['target'].format(
                    operations_dir=self.operations_dir, candname=self.candname,
                    beamnumber=f"{beamnumber:03d}"))
            destpath = Path(
                self.directory_structure['filterbank']['destination'].format(
                    candidates_dir=self.candidates_dir, candname=self.candname,
                    beamnumber=f"{beamnumber:03d}"))
            self.link_file(sourcepath, destpath)

        self.candparams['filterbank'] = str(destpath.parent)
        self.candparams['filfile_cand'] = self.directory_structure['filterbank']['destination'].format(
            candidates_dir=self.candidates_dir, candname=self.candname,
            beamnumber=f"{self.candparams['ibeam']+1:03d}")
        self.logger.info(f"Filterbank linked for {self.candname}.")

    def link_beamformer_weights(self) -> None:
        """Link beamformer weights to candidate directory.

        Links the weights applied in the real-time system at the candidate
        time.
        """
        self.logger.info(
            f"Linking beamformer weights to candidate directory for "
            f"{self.candname}.")

        beamformer_dir = Path(
            self.directory_structure['beamformer_weights']['target'].format(
                operations_dir=self.operations_dir))
        destdir = Path(
            self.directory_structure['beamformer_weights']['destination'].format(
                candidates_dir=self.candidates_dir, candname=self.candname))
        beamformer_name = find_beamformer_weights(
            self.candtime, beamformer_dir)

        self.logger.info(f"Found beamformerweights: {beamformer_name}")

        sourcepaths = beamformer_dir.glob(
            f"beamformer_weights_{beamformer_name}*.dat")

        subband_pattern = re.compile(r'sb\d\d')
        for sourcepath in sourcepaths:
            subband = subband_pattern.findall(sourcepath.name)
            destpath = destdir / sourcepath.name
            self.link_file(sourcepath, destpath)
            self.candparams[f'beamformer_weights_{subband}'] = destpath

        sourcepath = beamformer_dir.glob(
            f"beamformer_weights_{beamformer_name}*.yaml")
        destpath = destdir / sourcepath.name
        self.link_file(sourcepath, destpath)
        self.candparams['beamformer_weights'] = destpath

        self.logger.info(f"Beamformer weights linked for {self.candname}.")

    def link_hdf5_files(self, hours_to_save: int = 2) -> None:
        """Link hdf5 correlated data files to the candidates directory.

        Links all files within `hours_to_save`/2 hours of the candidate time.

        Parameters
        ----------
        hours_to_save : int
            Number of hours to save around the candidate.
        """
        date_format = '%Y-%m-%d'

        today = self.candtime.strftime(date_format)
        yesterday = (self.candtime - 1 * u.d).strftime(date_format)
        tomorrow = (self.candtime + 1 * u.d).strftime(date_format)
        start = self.candtime - hours_to_save / 2 * u.h
        stop = self.candtime + hours_to_save / 2 * u.h

        self.logger.info(
            f"Linking HDF5 files for {hours_to_save} hours to candidate "
            f"directory for {self.candname}.")

        source_dir = self.operations_dir / "correlator"
        sourcepaths = chain(
            source_dir.glob(f"{today}*hdf5"),
            source_dir.glob(f"{yesterday}*hdf5"),
            source_dir.glob(f"{tomorrow}*hdf5"))

        tokeep = []
        for sourcepath in sourcepaths:
            filetime = time_from_hdf5_filename(sourcepath)
            if within_times(start, stop, filetime):
                tokeep.append(sourcepath)

        destpath = Path(
            self.directory_structure['hdf5_files']['destination'].format(
                candidates_dir=self.candidates_dir, candname=self.candname))
        for sourcepath in tokeep:
            self.link_file(sourcepath, destpath / sourcepath.name)

        self.logger.info(
            f"{len(tokeep)} hdf5 files linked for {self.candname}.")

        self.candparams['hdf5_files'] = (
            self.directory_structure['hdf5_files']['destination'].format(
                candidates_dir=self.candidates_dir, candname=self.candname))

    def link_field_ms(self) -> None:
        """Link the field measurement at the time of the candidate."""
        raise NotImplementedError

    def link_caltables(self):
        """Link delay and bandpass calibration tables to the candidates directory.

        Links tables generated from the most recent calibrator observation
        prior to the candidate.
        """
        raise NotImplementedError

    def link_T2_csv(self):
        """Link the T2 csv file to the candidates directory."""

        self.logger.info(
            f"Linking T2 csv to candidate directory for {self.candname}.")

        sourcepath = Path(
            self.directory_structure['T2_csv']['target'].format(
                operations_dir=self.operations_dir))
        destpath = Path(
            self.directory_structure['T2_csv']['destination'].format(
                candidates_dir=self.candidates_dir, candname=self.candname))
        self.link_file(sourcepath, destpath)
        self.candparams['T2_csv'] = str(destpath)

        self.logger.info(
            f"Linked T2 csv to candidate directory for {self.candname}")

    def link_filplot_and_json(self):
        """Link the filplotter json and png files."""
        self.logger.info(
            f"Linking filplotter json and png to candidate directory for "
            f"{self.candname}.")

        for file in ['filplot_json', 'filplot_png']:
            sourcepath = Path(
                self.directory_structure[file]['target'].format(
                    operations_dir=self.operations_dir, candname=self.candname))
            destpath = Path(
                self.directory_structure[file].format(
                    candidates_dir=self.candidates_dir, candname=self.candname))
            self.link_file(sourcepath, destpath)

        self.candparams['filplot_cand'] = str(destpath)

        self.logger.info(
            f"Linked filplotter json and png to candidate directory for "
            f"{self.candname}")

    def link_file(self, sourcepath: Path, destpath: Path) -> None:
        """Link `destpath` to `sourcepath` if `sourcepath` does not already exist.

        Parameters
        ----------
        sourcepath : Path
            Path to link from.
        destpath : Path
            Path to link to.
        """
        try:
            sourcepath.link_to(destpath)
        except FileExistsError:
            self.logger.warning(
                f"{destpath} already exists. Skipped linking {sourcepath}.")
        else:
            self.logger.info(f"Linked {sourcepath} to {destpath}.")


def within_times(start_time: Time, end_time: Time, time: Time) -> bool:
    """Check if `time` is between `start_time` and `end_time`.

    Parameters
    ----------
    start_time : Time
        Start time of the interval.
    end_time : Time
        End time of the interval.
    time : Time
        Time to check if lies within the interval.

    Returns
    -------
    bool
        True if `time` is between `start_time` and `end_time`.
    """
    return start_time <= time <= end_time


def time_from_hdf5_filename(sourcepath: Path) -> Time:
    """Get time from hdf5 file name.

    Parameters
    ----------
    sourcepath : Path
        Path to hdf5 file.

    Returns
    -------
    Time
        Approximate start time of the file.
    """
    return Time(sourcepath.stem.split('_')[0])


def find_beamformer_weights(candtime: Time, bfdir: Path) -> str:
    """Find the beamformer weights that were in use at a time `candtime`.

    The times in the beamformer weight names are the times when they were
    uploaded to the correlator nodes. Therefore, we want the most recent
    calibration files that were created before `candtime`.

    Parameters
    ----------
    candtime : Time
        Time of the candidate.
    bfdir : Path
        Path to the beamformer weights directory.

    Returns
    -------
    str
        Name of the beamformer weights applied at `candtime`.
    """
    isot_string = (
        r"[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9]")
    isot_pattern = re.compile(isot_string)
    avail_calibs = sorted(
        [
            isot_pattern.findall(str(calib_path))[0] for calib_path
            in bfdir.glob(f"beamformer_weights_{isot_string}.yaml")],
        reverse=True)
    for avail_calib in avail_calibs:
        if avail_calib < isot_pattern.findall(candtime.isot)[0]:
            return avail_calib

    raise RuntimeError(f"No beamformer weights found for {candtime.isot}")
