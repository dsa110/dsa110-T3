import os
from pathlib import Path
from copy import deepcopy
from itertools import chain
from astropy.time import Time
from dsautils import dsa_syslogger as dsl
from dsautils import cnf


class DataManager:
    """Manage Level 1 data for confirmed candidates."""
    operations_dir = Path("/dataz/dsa110/operations")
    candidates_dir = Path("/dataz/dsa110/candidates")
    candidates_subdirs = [
        "Level3", "Level2/voltages", "Level2/filterbank",
        "Level2/calibration", "other"]
    
    subband_corrnames = list(cnf.Conf().get('corr')['ch0'].keys())
    nbeams = 256

    directory_structure = {
        'voltages':
            {
                'target': "{operations_dir}/T3/voltages/{hostname}_{candname}_data.out",
                'destination': "{candidates_dir}/{candname}/Level2/voltages/{candname}_{subband}_data.out",
            },
        'filterbank':
            {
                'target': "{operations_dir}/T1/{candname}/{candname}_{beamnumber}.fil",
                'destination': "{candidates_dir}/{candname}/Level2/filterbank/{candname}_{beamnumber}.fil",
            },
        'beamfomer_weights':
            {
                'target': "{operations_dir}/beamformer_weights/applied/{bfname}*",
                'destination': "{candidates_dir}/{candname}/Level2/calibration/"
            },
        'hdf5_files':
            {
                'target': "{operations_dir}/correlator/{hdf5_name}*.hdf5",
                'destination': "{candidates_dir}/{candname}/Level3/"
            },
        'T2_csv':
            {
                'target': "{operations_dir}/T2/cluster_output.csv",
                'destination': "{candidates_dir}/{candname}/Level2/T2_{candname}.csv"
            }
    }

    def __init__(self, candparams: dict, logger: dsl.DsaSyslogger = None):
        """Initialize info from candidate."""
        if logger is None:
            self.logger = dsl.DsaSyslogger()
        else:
            self.logger = logger

        self.candname = candparams['trigname']
        self.candparams = deepcopy(candparams)
    
    def __call__(self):
        self.create_directory_structure()
        return self.candparams

    def create_directory_structure(self):
        """Create directory structure for candidate."""

        self.logger.info("Creating directory structure for candidate.")
        self.cand_dir = self.candidates_dir / self.candname
        self.logger.info(f"Candidate directory: {self.cand_dir}")

        for subdir in self.candidates_subdirs:
            newdir = self.cand_dir / subdir
            if not newdir.exists():
                newdir.mkdir(parents=True)

        self.logger.info("Directory structure created.")

    def link_voltages(self):
        """Link voltages to candidate directory."""

        self.logger.info("Linking voltages to candidate directory.")

        for subband, corrname in enumerate(self.subband_corrnames):
            source_file = Path(
                self.directory_structure['voltages']['target'].format(
                    operations_dir=self.operations_dir, candname=self.candname,
                    hostname=corrname))
            dest_file = Path(
                self.directory_structure['voltages']['destination'].format(
                    candidates_dir=self.candidates_dir, candname=self.candname,
                    subband=f"sb{subband:02d}"))

            self.logger.info(f"Linking {source_file} to {dest_file}")
            dest_file.link_to(source_file)
            self.candparams[f'voltage_sb{subband:02d}'] = str(dest_file)

        self.logger.info("Voltages linked.")

    def link_filterbank(self):
        """Link filterbank to candidate directory."""

        self.logger.info("Linking filterbank to candidate directory.")

        for beamnumber in range(self.nbeams):
            source_file = Path(
                self.directory_structure['filterbank']['target'].format(
                    operations_dir=self.operations_dir, candname=self.candname,
                    beamnumber=beamnumber))
            dest_file = Path(
                self.directory_structure['filterbank']['destination'].format(
                    candidates_dir=self.candidates_dir, candname=self.candname,
                    beamnumber=beamnumber))
            self.logger.info(f"Linking {source_file} to {dest_file}")
            dest_file.link_to(source_file)

        if self.nbeams:     
            self.candparams['filterbank'] = dest_file.parent
            self.logger.info("Filterbank linked.")

    def link_beamformer_weights(self):
        pass

    def link_hdf5_files(self, hours_to_save: int = 2):
        candtime = Time(self.candparams['mjds'], 
        format='mjd', precision=0)
        start = candtime - hours_to_save / 2 * u.h
        stop = candtime + hours_to_save / 2 * u.h
        
        self.logger.info(f"Linking HDF5 files for {hours_to_save} hours.")
        
        

    def link_field_ms(self):
        pass

    def link_caltables(self):
        pass

    def link_T2_csv(self):
        """Link T2 csv to candidate directory."""

        self.logger.info("Linking T2 csv to candidate directory.")

        source_file = Path(
            self.directory_structure['T2_csv']['target'].format(
                operations_dir=self.operations_directory))
        dest_file = Path(
            self.directory_structure['T2_csv']['destination'].format(
                candidates_dir=self.candidates_dir, candname=self.candname))
        self.logger.info(f"Linking {source_file} to {dest_file}")
        dest_file.link_to(source_file)
        self.candparams['T2_csv'] = str(dest_file)
        self.logger.info("T2 csv linked.")


def within_times(start_time, end_time, time):
    """Check if time is within start and end times."""
    return start_time <= time <= end_time


def time_from_hdf5_filename(filepath):
    """Get time from hdf5 file name."""
    return Time(filepath.stem.split('_')[0])


