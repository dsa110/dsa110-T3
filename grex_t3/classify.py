import numpy as np 
import h5py
from keras.models import load_model

import analysis_tools

def classify(model, data):
    if type(model) == str:
        model = load_model(model)

    predictions = model.predict(data)

    return predictions

def prepare_dynspec(fn):
    data = analysis_tools.read_voltage(fn)
    I = analysis_tools.voltage_to_stokes(data)

    return I