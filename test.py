# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 13:46:40 2022

@author: AntoonGar

test script for the 4-maser engine.
another version of this repo exists for the study Dirac monopoles-induced peaks.
"""
import os
import sys

libPath = r"./"
sys.path.append(libPath)

from fourmaser_v5 import *
from psd_v1 import *

path = r"./"
os.chdir(path)

import numpy as np
import time
import pickle

#%%
save = True
saveDir = r"./"

verbose = 1

batchNb = 2
maxTime = 512
Fs = 1


# Seperate data into multiple training/test/validation batches.
# This prevent creating files larger than GPU VRAM, adjust at will
for ii in range(batchNb):
    print("-------------" + str(ii + 1) + "/" + str(batchNb) + "-------------")
    seed = round(time.time())

    #% training/test/validation data (filename) and (size :: nb Samples)
    if ii == 0:
        saveName = "ValidationData"
        numSamples = int(1e2)

    elif ii == 1:
        saveName = "TestData"
        numSamples = int(1e2)

    else:
        saveName = "TrainData_" + str(ii - 2)
        numSamples = int(1e2)

    # engine parameters: used to recreate your lab experiments. SI units!
    signalParamDic = {
        "freq": [0.05, 0.125],
        "phi": [0, 2 * np.pi],
        "tau": [0, 0],
        "fmod": [0.00390625, 0.01953125],
        "Imod": [0.0001, 0.0001],
        "noiseParams": [0.05, 0.4],
        "driftParam": 0.02,
        "lag": [0, 0],
        "Fs": Fs,
        "modTypes": ["FM"],
        "nbSamples": numSamples,
        "maxTime": maxTime,
        "sameImod": True,
        "binaryLag": False,
        "lagNb": 0,
        "sameLag": False,
        "amp_monopole": [1, 4],
        "alwaysEvent": True,
    }

    # data generation
    data_generator = fourmaser_v5(
        signalParams=signalParamDic, randomized=False, seed=seed, verbose=verbose
    )
    carrier, modulation, modulated, real, noise, drift, metadata = data_generator.dataGen()

    # power spectral density (not always needed)
    PSD_generator = psd_v1(real, Fs, "FM", verbose=0)
    psds_real, freq = PSD_generator.PSD()

    # save routine
    if save == True:
        if ii == 0:  # save user inputs during first cycle
            try:
                os.stat(saveDir)
            except:
                os.mkdir(saveDir)
                print("dir created")

            # generator parameter and metadata at time of generation.
            # usefull to avoid massive queries in the SQL db... :D
            fullDir = saveDir + "generator_parameters" + ".pickle"
            pickle.dump(signalParamDic, open(fullDir, "wb"))

            # saves a copy that can be opened manually using NPzee
            np.savez(fullDir + ".npz", **signalParamDic)

        # Data is saved as float32 in this dictionnary.
        DictToSave = {
            "carrier": np.float32(carrier),
            "modulation": np.float32(modulation),
            "modulated": np.float32(modulated),
            "real": np.float32(real),
            "freq": np.float32(freq),
            "psds_real": np.float32(psds_real),
            "samples_metadata": metadata["samples_metadata"],
            "batch_metadata": metadata["batch_metadata"],
        }

        fullDir = saveDir + saveName + ".pickle"
        pickle.dump(DictToSave, open(fullDir, "wb"))

        # del carrier, modulation, modulated, real, noise, drift, metadata, DictToSave
