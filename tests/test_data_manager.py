from pathlib import Path

import numpy as np
import astropy.units as u
from astropy.time import Time

from dsaT3.data_manager import (
    DataManager, find_beamformer_weights, time_from_hdf5_filename,
    within_times)

CANDPARAMS = {
    'trigname': 'test',
    'mjds': 57724.6234,
    'ibeam': 1
}
BFWEIGHTS_OFFSETS_DAYS = np.arange(-7, 2)+0.5
HDF5FILES_OFFSETS_HOURS = np.arange(-10, 10)+0.5


class FakeLogger:

    def __init__(self):
        pass

    def info(self, message):
        print(message)

    def warning(self, message):
        print(message)

    def error(self, message):
        print(message)


class FakeDataManager(DataManager):

    def __init__(self, candparams: dict, tmpdir: str):
        super().__init__(candparams, FakeLogger())

        self.operations_dir = Path(f"{tmpdir}/operations")
        self.candidates_dir = Path(f"{tmpdir}/candidates")
        self.voltage_dir = Path(f"{tmpdir}/T3")
        self.subband_corrnames = ('corr01', 'corr02', 'corr03')
        self.nbeams = 4

        self.fake_operations_data()

    def fake_operations_data(self):
        self.fake_voltage_data()
        self.fake_filterbank_data()
        self.fake_beamformerweights_data()
        self.fake_hdf5_data()
        self.fake_T2_data()
        self.fake_filplot_data()

    def fake_voltage_data(self):
        for corr in self.subband_corrnames:
            voltage_path = Path(
                self.directory_structure['voltages']['target'].format(
                    voltage_dir=self.voltage_dir, operations_dir=self.operations_dir, hostname=corr,
                    candname=self.candname))
            if not voltage_path.parent.exists():
                voltage_path.parent.mkdir(parents=True)
            voltage_path.touch()

    def fake_filterbank_data(self):
        filterbank_dir = Path(
            self.directory_structure['filterbank']['target'].format(
                operations_dir=self.operations_dir, beamnumber='XX',
                candname=self.candname)).parent
        if not filterbank_dir.exists():
            filterbank_dir.mkdir(parents=True)
        for beam in range(self.nbeams):
            filterbank_path = Path(
                self.directory_structure['filterbank']['target'].format(
                    operations_dir=self.operations_dir,
                    beamnumber=f"{beam:03d}",
                    candname=self.candname))
            filterbank_path.touch()

    def fake_beamformerweights_data(self):
        times = self.candtime + BFWEIGHTS_OFFSETS_DAYS * u.day

        bf_dir = Path(
            self.directory_structure['beamformer_weights']['target'].format(
                operations_dir=self.operations_dir))
        if not bf_dir.exists():
            bf_dir.mkdir(parents=True)

        for time in times:
            for sb in range(len(self.subband_corrnames)):
                bf_path = (
                    bf_dir / f"beamformer_weights_{time.isot}_sb{sb:02d}.dat")
                bf_path.touch()
            bf_path = bf_dir / f"beamformer_weights_{time.isot}.yaml"
            bf_path.touch()

    def fake_hdf5_data(self):
        times = self.candtime + HDF5FILES_OFFSETS_HOURS * u.hour

        hdf5_dir = Path(
            self.directory_structure['hdf5_files']['target'].format(
                operations_dir=self.operations_dir, hdf5_name='temp')).parent
        if not hdf5_dir.exists():
            hdf5_dir.mkdir(parents=True)

        for time in times:
            for sb in range(len(self.subband_corrnames)):
                hdf5_path = hdf5_dir / f"{time.isot}_sb{sb:02d}.hdf5"
                hdf5_path.touch()

    def fake_T2_data(self):
        T2_path = Path(
            self.directory_structure['T2_csv']['target'].format(
                operations_dir=self.operations_dir, candname=self.candname))
        if not T2_path.parent.exists():
            T2_path.parent.mkdir(parents=True)
        T2_path.touch()

    def fake_filplot_data(self):
        for file in ['filplot_json', 'filplot_png']:
            filplot_path = Path(
                self.directory_structure[file]['target'].format(
                    operations_dir=self.operations_dir,
                    candname=self.candname))
            if not filplot_path.parent.exists():
                filplot_path.parent.mkdir(parents=True)
            filplot_path.touch()


def test_datamanager_init(tmpdir):
    dm = FakeDataManager(CANDPARAMS, tmpdir)
    assert dm.logger is not None
    assert dm.candname == CANDPARAMS['trigname']
    assert dm.operations_dir == Path(f"{tmpdir}/operations")
    assert dm.candidates_dir == Path(f"{tmpdir}/candidates")


def test_datamanager_call(tmpdir):
    dm = FakeDataManager(CANDPARAMS, tmpdir)
    dm()

    for subdir in (
            "Level3", "Level2/voltages", "Level2/filterbank",
            "Level2/calibration", "other"):
        assert (dm.candidates_dir / dm.candname / subdir).exists()

    filterbank_path = Path(
        dm.directory_structure['filterbank']['destination'].format(
            candidates_dir=dm.candidates_dir, candname=dm.candname,
            beamnumber='*'))
    assert len(list(filterbank_path.parent.glob(
        filterbank_path.name))) == dm.nbeams

    beamformer_path = Path(
        dm.directory_structure['beamformer_weights']['destination'].format(
            candidates_dir=dm.candidates_dir, candname=dm.candname))
    assert len(list(beamformer_path.glob('beamformer_weights*.dat'))
               ) == len(dm.subband_corrnames)
    assert len(list(beamformer_path.glob('beamformer_weights*.yaml'))) == 1

    T2_path = Path(dm.directory_structure['T2_csv']['destination'].format(
        candidates_dir=dm.candidates_dir, candname=dm.candname))
    assert T2_path.exists()


def test_datamanager_create_directory_structure(tmpdir):
    dm = FakeDataManager(CANDPARAMS, tmpdir)
    dm.create_directory_structure()
    for subdir in (
            "Level3", "Level2/voltages", "Level2/filterbank",
            "Level2/calibration", "other"):
        assert (dm.candidates_dir / dm.candname / subdir).exists()


def test_datamanager_copy_voltages(tmpdir):
    dm = FakeDataManager(CANDPARAMS, tmpdir)
    dm.create_directory_structure()

    dm.copy_voltages()

    voltage_path = Path(
        dm.directory_structure['voltages']['destination'].format(
            candidates_dir=dm.candidates_dir, candname=dm.candname,
            subband='*'))
    assert len(list(voltage_path.parent.glob(voltage_path.name))
               ) == len(dm.subband_corrnames)

    for sb in range(len(dm.subband_corrnames)):
        assert isinstance(dm.candparams[f'voltage_sb{sb:02d}'], str)


def test_datamanager_link_filterbank(tmpdir):
    dm = FakeDataManager(CANDPARAMS, tmpdir)
    dm.create_directory_structure()
    dm.link_filterbank()

    filterbank_path = Path(
        dm.directory_structure['filterbank']['destination'].format(
            candidates_dir=dm.candidates_dir, candname=dm.candname,
            beamnumber='*'))
    assert len(list(filterbank_path.parent.glob(
        filterbank_path.name))) == dm.nbeams

    assert isinstance(dm.candparams["filterbank"], str)


def test_datamanager_link_beamformer_weights(tmpdir):
    dm = FakeDataManager(CANDPARAMS, tmpdir)
    dm.create_directory_structure()
    dm.link_beamformer_weights()

    beamformer_path = Path(
        dm.directory_structure['beamformer_weights']['destination'].format(
            candidates_dir=dm.candidates_dir, candname=dm.candname))
    assert len(list(beamformer_path.glob('beamformer_weights*.dat'))
               ) == len(dm.subband_corrnames)
    assert len(list(beamformer_path.glob('beamformer_weights*.yaml'))) == 1
    for subband in range(len(dm.subband_corrnames)):
        assert isinstance(dm.candparams[f"beamformer_weights_sb{subband:02d}"], str)
    assert isinstance(dm.candparams["beamformer_weights"], str)


def test_datamanager_link_hdf5_files(tmpdir):
    dm = FakeDataManager(CANDPARAMS, tmpdir)
    dm.create_directory_structure()
    dm.link_hdf5_files(filelength_min=60.)

    hdf5_path = Path(
        dm.directory_structure['hdf5_files']['destination'].format(
            candidates_dir=dm.candidates_dir, candname=dm.candname))
    assert len(list(hdf5_path.glob('*.hdf5'))) == len(dm.subband_corrnames)*2
    assert isinstance(dm.candparams["hdf5_files"], str)


def test_datamanager_link_field_ms(tmpdir):
    pass


def test_datamanager_link_caltables(tmpdir):
    pass


def test_datamanager_copy_T2csv(tmpdir):
    dm = FakeDataManager(CANDPARAMS, tmpdir)
    dm.create_directory_structure()

    dm.copy_T2_csv()

    T2_path = Path(dm.directory_structure['T2_csv']['destination'].format(
        candidates_dir=dm.candidates_dir, candname=dm.candname))
    assert T2_path.exists()
    assert isinstance(dm.candparams["T2_csv"], str)


def test_link_filplot_and_json(tmpdir):
    dm = FakeDataManager(CANDPARAMS, tmpdir)
    dm.create_directory_structure()

    dm.link_filplot_and_json()

    for file in ['filplot_json', 'filplot_png']:
        filepath = Path(dm.directory_structure[file]['destination'].format(
            candidates_dir=dm.candidates_dir, candname=dm.candname))
        assert filepath.exists()
    assert isinstance(dm.candparams['filplot_cand'], str)


def test_datamanager_link_file(tmpdir):
    dm = FakeDataManager(CANDPARAMS, tmpdir)
    dm.create_directory_structure()

    destpath = Path(tmpdir) / "dest.txt"
    sourcepath = Path(tmpdir) / "source.txt"
    sourcepath.touch()

    dm.link_file(sourcepath, destpath)
    assert destpath.exists()
    assert not destpath.is_symlink()
    assert destpath.is_file()

    # Check that already existing paths are handled correctly
    dm.link_file(sourcepath, destpath)


def test_within_times():
    time0 = Time('2020-01-01T23:30:00')
    times = time0 + np.arange(3) * u.hour
    assert within_times(times[0], times[2], times[1])
    assert not within_times(times[0], times[1], times[2])
    assert not within_times(times[1], times[2], times[0])


def test_time_from_hdf5_filename():
    outtime = time_from_hdf5_filename(Path(
        '/some/path/to/a/directory/2022-01-01T03:23:45_sb00.hdf5'))
    assert abs((outtime - Time('2022-01-01T03:23:45')).to_value(u.s)) < 1e-5


def test_find_beamformer_weights(tmpdir):
    dm = FakeDataManager(CANDPARAMS, tmpdir)
    beamformer_dir = Path(
        dm.directory_structure['beamformer_weights']['target'].format(
            operations_dir=dm.operations_dir))
    found_weights = find_beamformer_weights(dm.candtime, beamformer_dir)
    assert abs((dm.candtime - Time(found_weights)).to_value(u.d) - 0.5) < 1e-5
