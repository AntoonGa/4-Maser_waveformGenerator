# -*- coding: utf-8 -*-
"""
Created on 08 feb 2021

this class grabs time domain signals and outputs their PSD
the PSD normalization is done within the script

"""

import matplotlib.pyplot as plt
import numpy as np
import random
import time
import bottleneck as bn
from tqdm import tqdm
import math
from numpy import random
import scipy.special
import traceback
from scipy import fftpack
from numpy.fft import fft


# from scipy.fft import fft, fftfreq
# # Number of sample points
# N = 600
# # sample spacing
# T = 1.0 / 800.0
# x = np.linspace(0.0, N*T, N, endpoint=False)
# y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
# yf = fft(y)
# xf = fftfreq(N, T)[:N//2]
# import matplotlib.pyplot as plt
# plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
# plt.grid()
# plt.show()


class psd_v1:
    def __init__(self, timeSeries, Fs=1, modType="Fm", verbose=0):

        self.modType = modType
        self.timeSeries = timeSeries
        self.Fs = Fs
        # self.dt = 1/self.Fs
        self.N = np.shape(timeSeries)[-1]
        # self.timeAxis =  np.linspace(0.0, self.N*self.dt, self.N, endpoint=False)

        self.dataShape = np.shape(np.shape(self.timeSeries))

        self.verbose = verbose

    def PSD(self):
        tt1 = time.time()

        nbSample = np.shape(self.timeSeries)[0]

        x = self.timeSeries

        PSDs = [
            [
                np.fft.fft(x[ii, 0, :] - np.mean(x[ii, 0, :])),
                np.fft.fft(x[ii, 1, :] - np.mean(x[ii, 1, :])),
                np.fft.fft(x[ii, 2, :] - np.mean(x[ii, 2, :])),
                np.fft.fft(x[ii, 3, :] - np.mean(x[ii, 3, :])),
            ]
            for ii in range(nbSample)
        ]

        PSDs = np.array(PSDs)
        PSDs = PSDs[:, :, 0 : int(self.N / 2) + 1]
        PSDs = np.abs(PSDs) ** 2 / (
            self.N**2
        )  # amplitude is independent of aqcuisition length

        freqaxis = [ii * self.Fs / self.N for ii in range(self.N)]
        freqaxis = freqaxis[0 : int(self.N / 2) + 1]

        # normalization
        if self.modType == "FM":
            PSDs = PSDs * 100 + 0.1
        elif self.modType == "AM":
            PSDs = PSDs * 100 + 0.1
        elif self.modType == "POLY":
            PSDs = PSDs * 100 + 0.1

        if self.verbose != 0:
            plt.figure(0)
            plt.hist(PSDs.flatten(), bins=int(self.N / 50))
            plt.title("PSD, Data distribution")
            plt.yscale("log")
            plt.show()

            countGreaterThanOne = (PSDs.flatten() > 1.0).sum()

            print("_______________PSD________________")
            print(
                "Elements greater than one (noisy): "
                + str(countGreaterThanOne)
                + "/"
                + str(len(PSDs.flatten()))
            )
            print("Max value (noisy): " + str(max(PSDs.flatten())))
            print("Min value (noisy): " + str(min(PSDs.flatten())))
            print("__________________________________")

            self.fastVerbose(PSDs, freqaxis, self.verbose)

        dispstring = str(round(time.time() - tt1, 2))
        print("__________________________________")
        print("PSDs ready, in: " + dispstring + " seconds")
        print("Shape PSDs: " + str(np.shape(PSDs)))
        print("__________________________________")

        return PSDs, freqaxis

    def fastVerbose(self, PSDs, freqaxis, nbSample):

        for jj in range(nbSample):
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True)
            fig.suptitle("Sample: " + str(jj + 1))
            ax1.plot(freqaxis, PSDs[jj, 0, :])
            ax2.plot(freqaxis, PSDs[jj, 1, :])
            ax3.plot(freqaxis, PSDs[jj, 2, :])
            ax4.plot(freqaxis, PSDs[jj, 3, :])
            plt.show()
