import numpy as np
import xarray
import matplotlib.pyplot as plt
import pandas as pd


def read_voltage_data(file_name, timedownsample=None,
                      freqdownsample=None):
    """ Read in the voltage data from a .nc file 
    and return it as StokesI.
    """
    ds = xarray.open_dataset(file_name, chunks={"time": 1000})

    # Create complex numbers from Re/Im
    voltages = ds["voltages"].sel(reim="real") + ds["voltages"].sel(reim="imaginary") * 1j
    # Make Stokes I by converting to XX/YY and then creating XX**2 + YY**2
    stokesi = np.square(np.abs(voltages)).astype('int32')
    stokesi = stokesi.sum(dim='pol').astype('int32')  # Summing and then converting type if necessary

    # Compute in parallel (if using Dask)
    stokesi = stokesi.compute()

    if timedownsample is not None:
        stokesi = stokesi.coarsen(time=int(timedownsample), boundary='trim').mean()
    if freqdownsample is not None:
        stokesi = stokesi.coarsen(freq=int(freqdownsample), boundary='trim').mean()

    return stokesi