# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 14:19:57 2022

@author: grab

4-maser waveform generator

Input in the form a a dictionnary are all made MANDATORY for obvious reasons. Input shape is as follow:
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

call dataGen() to output:
    carrier, modulation, modulated, real, noise, drift, metadata, tag, dataRanges_ouputs, normalRanges_ouputs

# __init__ params are as following:
    
    #these are constrained in a certain range by the integration time and dt.
    #inputing values outside de range with narrow them down to the range
    #TURN OFF DECAY by setting TAU to 0
    #TURN OFF MODULATION by setting IMOD to 0
    
    #The signal are: frequency modulation, amplitude modulation or double sinwaves interference
    #On top of which Gaussian noise and drifts are linearly added
    #The normalization is strong enough that most values live between 0.3 to 0.7, suitable for most ML practice
    #The gaussian noise variance is picked up from the range each iteration of a signal
    #The drift is somewhat constant along the entire databatch, however this does not mean it is the same ! That is because the drift is generated via a pseudo-random-elastic walk
    
    #Randomized = True will insure all the variables are picked randomly from the widest range of parameters, usefull for general training
    
    #Fs (sampling frequency) is 1 sec, dt is 1sec, therefore the signal length is the size of the signal, this can be adjusted in the inputs
    #The outputShape input insures automatic conversion to the neural network shape : outputShape = ( nBsamples, sampleLength, virtual dim = 1, virtual dim= 1  ), suitable for Tensorflow 1D/2DCNN
       
    # metaData has the following shape:
    # [freq,phi,fmod,Imod,tau, noiseSigma, noiseDrift]
    
    # The signals are outputted accross 4 channels, simulating 4-MASER outputing data simultaneously.
    # MASER Channel can be turned off induvidually
    # MASER Channel can be laggy relatvie to one an other, simulating a propagating event accross the 4-MASER
    
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


class fourmaser_v5:
    def __init__(self, signalParams, randomized=True, seed=None, verbose=1):
        # seed value
        self.seed = seed
        # verbosity
        self.verbose = verbose

        # Intersample configuration
        self.modTypes = signalParams["modTypes"]
        # Full set configuration
        self.total_num_samples = int(signalParams["nbSamples"])
        self.randomized = randomized
        # Intrasample configuration
        self.Fs = signalParams["Fs"]
        self.dt = 1 / self.Fs
        self.maxTime = signalParams["maxTime"]
        self.sample_size = int(self.maxTime * self.Fs)
        self.starting_point = 0
        # Set modulation index identical accross masers
        self.sameImod = signalParams["sameImod"]
        # Set lag to be zero 50% of the time (for classification)
        self.binaryLag = signalParams["binaryLag"]
        # Set number of different lags vs. first one
        self.lagNb = signalParams["lagNb"]
        # Set lag to be the same
        self.sameLag = signalParams["sameLag"]

        # assign user param in dict
        self.userParams = signalParams

        # signal and noise params physical limitations
        self.amp = 1  # amplitude is always 1, noise is the scale parameters
        self.fMax = (
            self.Fs / 8
        )  # 8 samples per oscillation, more stringent than the usual Nqyst
        self.fMin = 10 / self.maxTime  # 10 full oscillation per measurement
        self.tauMin = 0.2  # this is in units of total time! (exponential decay)
        self.tauMax = 8
        self.fmodMin = (
            2 / self.maxTime
        )  # 1 full oscillation per measurement        #fmod ranges
        self.fmodMax = self.fMin  # Always smaller than the carrier
        self.ImodMin = 0  # modulation index, unitless.
        self.ImodMax = 1
        self.noiseMin = 0  # Gaussian noise (linear combinaison with signal, this is the coeff)
        self.noiseMax = 1
        self.driftMin = 0  # NOT USED; you can input any value ! random walk added to the noise linearly with that coef.
        self.driftMax = 1
        self.lagMin = 0
        self.lagMax = self.maxTime

        # assign signal generation param range: this insures the input range is withing physical boundaries
        self.freqRange = [
            min(max(signalParams["freq"][0], self.fMin), self.fMax),
            max(min(signalParams["freq"][1], self.fMax), self.fMin),
        ]
        self.phiRange = [
            min(max(signalParams["phi"][0], 0), np.pi * 2),
            max(min(signalParams["phi"][1], np.pi * 2), 0),
        ]
        self.tauRange = [
            min(max(signalParams["tau"][0], self.tauMin), self.tauMax),
            max(min(signalParams["tau"][1], self.tauMax), self.tauMin),
        ]
        self.fmodRange = [
            min(max(signalParams["fmod"][0], self.fmodMin), self.fmodMax),
            max(min(signalParams["fmod"][1], self.fmodMax), self.fmodMin),
        ]
        self.ImodRange = [
            min(max(signalParams["Imod"][0], self.ImodMin), self.ImodMax),
            max(min(signalParams["Imod"][1], self.ImodMax), self.ImodMin),
        ]
        self.lagRange = [
            min(max(signalParams["lag"][0], self.lagMin), self.lagMax),
            max(min(signalParams["lag"][1], self.lagMax), self.lagMin),
        ]
        # Correct meta actual Values, zero turns off the effect
        if self.ImodRange == [0, 0]:
            self.fmodRange = [0, 0]

        if signalParams["tau"][0] == 0 or signalParams["tau"][1] == 0:
            self.tauRange = [0, 0]

        # noise param param range: this insures the input range is withing boundaries
        noiseParams = np.sort(signalParams["noiseParams"])
        self.noiseRange = [
            min(max(noiseParams[0], self.noiseMin), self.noiseMax),
            max(min(noiseParams[1], self.noiseMax), self.noiseMin),
        ]
        self.drift = signalParams["driftParam"]

        # Display signal and noise ranges during data generation
        if randomized == True:
            # assign signal param range: this insures the input range is withing boundaries
            self.freqRange = [self.fMin, self.fMax]
            self.phiRange = [0, np.pi * 2]
            self.tauRange = [self.tauMin, self.tauMax]
            self.fmodRange = [self.fmodMin, self.fmodMax]
            self.ImodRange = [self.ImodMin, self.ImodMax]
            self.lagRange = [self.lagMin, self.lagMax]
            print("_____________Default Ranges___________")
        else:
            print("________________Ranges________________")
        print("freqRange: " + str(self.freqRange))
        print("phiRange:  " + str(self.phiRange))
        print("tauRange:  " + str(self.tauRange))
        print("fmodRange: " + str(self.fmodRange))
        print("ImodRange: " + str(self.ImodRange))
        print("lagRange:  " + str(self.lagRange))
        print("____________Allowed Ranges____________")
        print("freqRange: " + str([self.fMin, self.fMax]))
        print("phiRange:  " + str([0, np.pi * 2]))
        print("tauRange:  " + str([self.tauMin, self.tauMax]))
        print("fmodRange: " + str([self.fmodMin, self.fmodMax]))
        print("ImodRange: " + str([self.ImodMin, self.ImodMax]))
        print("lagRange:  " + str([self.lagMin, self.lagMax]))
        print("______________Noise Param______________")
        print("noiseRange:" + str(self.noiseRange))
        print("drift:      " + str(self.drift))
        print("_______________________________________")
        print("______________Batch Param______________")
        print("sameImod:" + str(self.sameImod))
        print("binaryLag:" + str(self.binaryLag))
        print("lagNb:" + str(self.lagNb))
        print("sameLag:" + str(self.sameLag))
        print("Fs:" + str(self.Fs))
        print("dt:" + str(self.dt))
        print("maxTime:" + str(self.maxTime))
        print("sample_size:" + str(self.sample_size))
        print("starting_point:" + str(self.starting_point))
        print("_______________________________________")

        time.sleep(0.5)

        # Generate range arrays of output, and normalization ranges (physical boundaries), these are in the metadata output
        self.dataRanges_ouputs = {
            "freqRange": self.freqRange,
            "phiRange": self.phiRange,
            "fmodRange": self.fmodRange,
            "ImodRange": self.ImodRange,
            "tauRange": self.tauRange,
            "noiseRange": self.noiseRange,
            "lagRange": self.lagRange,
        }

        self.normalRanges_ouputs = {
            "freqRange": [self.fMin, self.fMax],
            "phiRange": [0, 2 * np.pi],
            "fmodRange": [self.fmodMin, self.fmodMax],
            "ImodRange": [self.ImodMin, self.ImodMax],
            "tauRange": [self.tauMin, self.tauMax],
            "noiseRange": [self.noiseMin, self.noiseMax],
            "lagRange": [self.lagMin, self.lagMax],
        }

    def dataGen(self):
        # Main function to call: waveforms and noise generation function. Then call displays.
        timeStamp1 = time.time()

        # carrier, modulation, modulated, real, noise, drift, metadata, tag = self.waveforms()
        carrier, modulation, modulated, real, noise, drift, metadata = self.waveforms()

        # #Shuffle and display
        # self.shuffle(    carrier, modulation, modulated, noise, drift, real, metadata,tag)
        # self.displayFunc(carrier, modulation, modulated, noise, drift, real, metadata,tag)

        timeStamp2 = time.time()
        dispstring = str(round(timeStamp2 - timeStamp1, 2))
        print("__________________________________")
        print("Data ready, in: " + dispstring + " seconds")
        print("Shape signals: " + str(np.shape(carrier)))
        print("keys Meta: " + str(metadata.keys()))
        print("__________________________________")

        if self.verbose > 0:
            self.fastVerbose(carrier, modulation, modulated, real, noise, drift, metadata)

        # return carrier, modulation, modulated, real, noise, drift, metadata, metaTags
        return carrier, modulation, modulated, real, noise, drift, metadata

    def waveforms(self):
        # Generating pure waveforms by calling respective function and rescale

        # Containers for samples and subsamples
        carrier = []
        modulation = []
        modulated = []
        metadata = []

        numSamplePerModType = int(self.total_num_samples / len(self.modTypes))

        if "FM" in self.modTypes:
            carrier_temp, modulation_temp, modulated_temp, metadata_temp = self.FMforms(
                numSamplePerModType
            )

            # append data
            carrier.append(carrier_temp)
            modulation.append(modulation_temp)
            modulated.append(modulated_temp)
            metadata.append(metadata_temp)

        carrier = np.squeeze(carrier)
        modulation = np.squeeze(modulation)
        modulated = np.squeeze(modulated)

        # Generate Gaussian Noise over all batch
        noise = self.gaussianNoise(np.shape(carrier), metadata)
        # #Generate Drifts over all batch
        drift = self.driftNoise(np.shape(noise))
        # Real samples
        real = np.add(modulated, np.add(noise, drift))

        # Rescale data
        # this insures the noisy data is largely between 0 and 1. All other fields are normalized the same way
        scale = 8
        carrier = np.array(carrier) / scale + 0.5
        modulation = np.array(modulation) / scale + 0.5
        modulated = np.array(modulated) / scale + 0.5
        real = np.array(real) / scale + 0.5
        noise = np.array(noise) / scale + 0.5
        drift = np.array(drift) / scale + 0.5

        # generating metadata dictionnary, adding up ranges
        batch_metadata = {
            "userInputRanges": self.dataRanges_ouputs,
            "normalizationRanges": self.normalRanges_ouputs,
            "userConfiguration": self.userParams,
        }

        metadata = {"samples_metadata": metadata[0], "batch_metadata": batch_metadata}

        return carrier, modulation, modulated, real, noise, drift, metadata

    def FMforms(self, numSamplePerModType):
        # FM Modulated sinewaves
        random.seed(self.seed)

        # placeholder for signals and modulations
        carrier_out = []
        modulation_out = []
        modulated_out = []
        metadata_out = []

        for j in tqdm(range(0, numSamplePerModType), desc="Generating Carriers and FM mods: "):
            # draw single instance of latent param
            (
                freq,
                phi1,
                Imod1,
                fmod,
                phimod,
                tau1,
                noiseSigma1,
                noiseDrift1,
                _,
            ) = self.drawLatentParams(random.randint(0, 5000))
            lag1 = 0  # first maser is the time reference, lag only happens on the modulation function
            # draw other 3 masers latent params:<random int to change the seed. Main seed still constrict choices.
            _, phi2, Imod2, _, _, tau2, noiseSigma2, noiseDrift2, lag2 = self.drawLatentParams(
                random.randint(0, 5000)
            )
            _, phi3, Imod3, _, _, tau3, noiseSigma3, noiseDrift3, lag3 = self.drawLatentParams(
                random.randint(0, 5000)
            )
            _, phi4, Imod4, _, _, tau4, noiseSigma4, noiseDrift4, lag4 = self.drawLatentParams(
                random.randint(0, 5000)
            )

            if self.sameImod == True:
                Imod2 = Imod1
                Imod3 = Imod1
                Imod4 = Imod1

            # All laggy masers have the same lag
            if self.sameLag == True:
                lag3 = lag2
                lag4 = lag2

            # Number of masers to lag
            if self.lagNb == 0:  # 0 Masers lags
                lag2 = lag1
                lag3 = lag1
                lag4 = lag1

            elif self.lagNb == 1:  # 1 Masers lags, choosen randomly
                lag2, lag3, lag4 = np.random.permutation([lag2, lag1, lag1])

            elif self.lagNb == 2:  # 2 Masers lags, choosen randomly
                lag2, lag3, lag4 = np.random.permutation([lag2, lag3, lag1])

            # 3 Masers lags is default

            # 50/50 chance of suppressing lag (for classification problems)
            if self.binaryLag == True:
                if random.uniform(0, 1) >= 0.5:  # 50/50 chance of having no lag
                    lag2 = lag1
                    lag3 = lag1
                    lag4 = lag1

            # flag for ease of classification
            if lag1 == 0 and lag2 == 0 and lag3 == 0 and lag4 == 0:
                lagFlag = 0
            else:
                lagFlag = 1

            # Generate sinewave
            if tau1 > 0 and tau2 > 0 and tau3 > 0 and tau4 > 0:
                carrier_temp1 = [
                    1
                    * math.cos(2 * np.pi * ii * self.dt * freq + phi1)
                    * math.exp(-ii * self.dt / (tau1 * self.maxTime))
                    for ii in range(self.sample_size)
                ]
                carrier_temp2 = [
                    1
                    * math.cos(2 * np.pi * ii * self.dt * freq + phi2)
                    * math.exp(-ii * self.dt / (tau2 * self.maxTime))
                    for ii in range(self.sample_size)
                ]
                carrier_temp3 = [
                    1
                    * math.cos(2 * np.pi * ii * self.dt * freq + phi3)
                    * math.exp(-ii * self.dt / (tau3 * self.maxTime))
                    for ii in range(self.sample_size)
                ]
                carrier_temp4 = [
                    1
                    * math.cos(2 * np.pi * ii * self.dt * freq + phi4)
                    * math.exp(-ii * self.dt / (tau4 * self.maxTime))
                    for ii in range(self.sample_size)
                ]

            else:
                tau1 = 0
                tau2 = 0
                tau3 = 0
                tau4 = 0
                carrier_temp1 = [
                    1 * math.cos(2 * np.pi * ii * self.dt * freq + phi1)
                    for ii in range(self.sample_size)
                ]
                carrier_temp2 = [
                    1 * math.cos(2 * np.pi * ii * self.dt * freq + phi2)
                    for ii in range(self.sample_size)
                ]
                carrier_temp3 = [
                    1 * math.cos(2 * np.pi * ii * self.dt * freq + phi3)
                    for ii in range(self.sample_size)
                ]
                carrier_temp4 = [
                    1 * math.cos(2 * np.pi * ii * self.dt * freq + phi4)
                    for ii in range(self.sample_size)
                ]

            # generate modulation
            if Imod1 != 0 and Imod2 != 0 and Imod3 != 0 and Imod4 != 0:
                modulatorSens = 0.0025
                mindex1 = modulatorSens * Imod1 / fmod
                mindex2 = modulatorSens * Imod2 / fmod
                mindex3 = modulatorSens * Imod3 / fmod
                mindex4 = modulatorSens * Imod4 / fmod
            else:
                mindex1 = 0
                mindex2 = 0
                mindex3 = 0
                mindex4 = 0

            modulation_temp1 = [
                Imod1 * math.cos(2 * np.pi * (ii * self.dt + lag1) * fmod + phimod)
                for ii in range(self.sample_size)
            ]
            modulation_temp2 = [
                Imod2 * math.cos(2 * np.pi * (ii * self.dt + lag2) * fmod + phimod)
                for ii in range(self.sample_size)
            ]
            modulation_temp3 = [
                Imod3 * math.cos(2 * np.pi * (ii * self.dt + lag3) * fmod + phimod)
                for ii in range(self.sample_size)
            ]
            modulation_temp4 = [
                Imod4 * math.cos(2 * np.pi * (ii * self.dt + lag4) * fmod + phimod)
                for ii in range(self.sample_size)
            ]

            # generate FM signal
            if tau1 > 0 and tau2 > 0 and tau3 > 0 and tau4 > 0:
                modulated_temp1 = [
                    1
                    * math.cos(
                        2 * np.pi * ii * self.dt * freq
                        + 2
                        * np.pi
                        * mindex1
                        * math.cos(2 * np.pi * (ii * self.dt + lag1) * fmod + phimod)
                        + phi1
                    )
                    * math.exp(-ii * self.dt / (tau1 * self.maxTime))
                    for ii in range(self.sample_size)
                ]
                modulated_temp2 = [
                    1
                    * math.cos(
                        2 * np.pi * ii * self.dt * freq
                        + 2
                        * np.pi
                        * mindex2
                        * math.cos(2 * np.pi * (ii * self.dt + lag2) * fmod + phimod)
                        + phi2
                    )
                    * math.exp(-ii * self.dt / (tau2 * self.maxTime))
                    for ii in range(self.sample_size)
                ]
                modulated_temp3 = [
                    1
                    * math.cos(
                        2 * np.pi * ii * self.dt * freq
                        + 2
                        * np.pi
                        * mindex3
                        * math.cos(2 * np.pi * (ii * self.dt + lag3) * fmod + phimod)
                        + phi3
                    )
                    * math.exp(-ii * self.dt / (tau3 * self.maxTime))
                    for ii in range(self.sample_size)
                ]
                modulated_temp4 = [
                    1
                    * math.cos(
                        2 * np.pi * ii * self.dt * freq
                        + 2
                        * np.pi
                        * mindex4
                        * math.cos(2 * np.pi * (ii * self.dt + lag4) * fmod + phimod)
                        + phi4
                    )
                    * math.exp(-ii * self.dt / (tau4 * self.maxTime))
                    for ii in range(self.sample_size)
                ]
            else:
                modulated_temp1 = [
                    1
                    * math.cos(
                        2 * np.pi * ii * self.dt * freq
                        + 2
                        * np.pi
                        * mindex1
                        * math.cos(2 * np.pi * (ii * self.dt + lag1) * fmod + phimod)
                        + phi1
                    )
                    for ii in range(self.sample_size)
                ]
                modulated_temp2 = [
                    1
                    * math.cos(
                        2 * np.pi * ii * self.dt * freq
                        + 2
                        * np.pi
                        * mindex2
                        * math.cos(2 * np.pi * (ii * self.dt + lag2) * fmod + phimod)
                        + phi2
                    )
                    for ii in range(self.sample_size)
                ]
                modulated_temp3 = [
                    1
                    * math.cos(
                        2 * np.pi * ii * self.dt * freq
                        + 2
                        * np.pi
                        * mindex3
                        * math.cos(2 * np.pi * (ii * self.dt + lag3) * fmod + phimod)
                        + phi3
                    )
                    for ii in range(self.sample_size)
                ]
                modulated_temp4 = [
                    1
                    * math.cos(
                        2 * np.pi * ii * self.dt * freq
                        + 2
                        * np.pi
                        * mindex4
                        * math.cos(2 * np.pi * (ii * self.dt + lag4) * fmod + phimod)
                        + phi4
                    )
                    for ii in range(self.sample_size)
                ]
            # Reshape
            carrier_temp = np.array(
                [carrier_temp1, carrier_temp2, carrier_temp3, carrier_temp4]
            )
            modulation_temp = np.array(
                [modulation_temp1, modulation_temp2, modulation_temp3, modulation_temp4]
            )
            modulated_temp = np.array(
                [modulated_temp1, modulated_temp2, modulated_temp3, modulated_temp4]
            )

            carrier_out.append(carrier_temp)
            modulation_out.append(modulation_temp)
            modulated_out.append(modulated_temp)

            meta_i = {
                "freq": [freq, freq, freq, freq],
                "phi": [phi1, phi2, phi3, phi4],
                "fmod": [fmod, fmod, fmod, fmod],
                "Imod": [Imod1, Imod2, Imod3, Imod4],
                "phimod": [phimod, phimod, phimod, phimod],
                "tau": [tau1, tau2, tau3, tau4],
                "noiseSigma": [noiseSigma1, noiseSigma2, noiseSigma3, noiseSigma4],
                "noiseDrift": [noiseDrift1, noiseDrift2, noiseDrift3, noiseDrift4],
                "lag": [lag1, lag2, lag3, lag4],
                "lagFlag": lagFlag,
                "sameImod": self.sameImod,
                "modType": "FM, modulatorSens: " + str(modulatorSens),
            }

            metadata_out.append(meta_i)

        return carrier_out, modulation_out, modulated_out, metadata_out

    def drawLatentParams(self, j):
        # Generate random parameters
        random.seed(self.seed + j)
        # Sinewave, modulation and decay parameters
        freq = random.uniform(self.freqRange[0], self.freqRange[1])
        phi = random.uniform(self.phiRange[0], self.phiRange[1])
        phimod = random.uniform(0, 2 * np.pi)
        Imod = random.uniform(self.ImodRange[0], self.ImodRange[1])
        tau = random.uniform(self.tauRange[0], self.tauRange[1])
        noiseSigma = random.uniform(self.noiseRange[0], self.noiseRange[1])
        noiseDrift = self.drift
        lag = random.uniform(self.lagRange[0], self.lagRange[1])

        # fmod is always slower than the carrier
        fmod = random.uniform(self.fmodRange[0], self.fmodRange[1])

        return freq, phi, Imod, fmod, phimod, tau, noiseSigma, noiseDrift, lag

    def gaussianNoise(self, theShape, metadata):
        # generate Gaussian noise.
        random.seed(self.seed)

        timeStamp1 = time.time()
        y_out = []
        nbSample = theShape[0]
        nbPoint = theShape[2]

        for ii in range(nbSample):
            sigmas = metadata[0][ii]["noiseSigma"]
            noise1 = np.random.normal(0, sigmas[0], nbPoint)
            noise2 = np.random.normal(0, sigmas[1], nbPoint)
            noise3 = np.random.normal(0, sigmas[2], nbPoint)
            noise4 = np.random.normal(0, sigmas[3], nbPoint)
            y_out.append([noise1, noise2, noise3, noise4])

        timeStamp2 = time.time()
        dispstring = str(round(timeStamp2 - timeStamp1, 2))
        print("Generated Gaussian Noise: " + dispstring + " seconds")
        time.sleep(0.5)
        return np.array(y_out)

    def driftNoise(self, theShape):
        # generating drift via a pseudo random walk
        time.sleep(0.5)
        timeStamp1 = time.time()

        random.seed(self.seed)

        # This is a random walk on even indexes, smoothed via moving average and then lineraly added to noise.
        # drift multiplier parameters
        driftToNoiseRatio = self.drift  # multiplier when adding drift to noise
        if driftToNoiseRatio > 0:
            # init arrays
            xwalk = np.zeros(theShape)
            smoothWalk = []

            # random walk Wiergner model
            for ii in tqdm(
                range(theShape[0]), desc="Generating Drifts:     "
            ):  # loop through sample number
                for kk in range(theShape[1]):  # loop through masers:

                    steps = random.choice((-1, 0, +1), theShape[2])

                    for jj in range(theShape[2]):
                        if jj % 2 == 0:  # Random everyEvenPoints
                            xwalk[ii, kk, jj] = xwalk[ii, kk, jj - 1] + steps[jj]
                        else:
                            xwalk[ii, kk, jj] = xwalk[ii, kk, jj - 1]

                # Smoothing and normalizing
                temp1 = bn.move_mean(xwalk[ii, 0, :], window=int(theShape[1] / 2), min_count=1)
                temp2 = bn.move_mean(xwalk[ii, 1, :], window=int(theShape[1] / 2), min_count=1)
                temp3 = bn.move_mean(xwalk[ii, 2, :], window=int(theShape[1] / 2), min_count=1)
                temp4 = bn.move_mean(xwalk[ii, 3, :], window=int(theShape[1] / 2), min_count=1)

                temp1 = temp1 / np.max(abs(temp1))
                temp2 = temp2 / np.max(abs(temp2))
                temp3 = temp3 / np.max(abs(temp3))
                temp4 = temp4 / np.max(abs(temp4))

                temp1 = temp1 * driftToNoiseRatio
                temp2 = temp2 * driftToNoiseRatio
                temp3 = temp3 * driftToNoiseRatio
                temp4 = temp4 * driftToNoiseRatio

                smoothWalk.append([temp1, temp2, temp3, temp4])

            timeStamp2 = time.time()
            dispstring = str(round(timeStamp2 - timeStamp1, 2))
            print("Generated Drifts: " + dispstring + " seconds")

        else:
            print("No drifts...")
            smoothWalk = np.zeros(theShape)

        print(np.shape(smoothWalk))

        return np.array(smoothWalk)

    def fastVerbose(self, carrier, modulation, modulated, real, noise, drift, metadata):
        # A quick and dirty display function, called after data generation.
        timeAxis = [ii * self.dt for ii in range(self.sample_size)]

        for target in range(self.verbose):
            fig, axs = plt.subplots(4, sharex=True, sharey=True)
            fig.suptitle("Carrier, Sample: " + str(target))
            axs[0].plot(timeAxis, carrier[target, 0, :])
            axs[1].plot(timeAxis, carrier[target, 1, :])
            axs[2].plot(timeAxis, carrier[target, 2, :])
            axs[3].plot(timeAxis, carrier[target, 3, :])

            fig, axs = plt.subplots(4, sharex=True, sharey=True)
            fig.suptitle("Modulation, Sample: " + str(target))
            axs[0].plot(timeAxis, modulation[target, 0, :])
            axs[1].plot(timeAxis, modulation[target, 1, :])
            axs[2].plot(timeAxis, modulation[target, 2, :])
            axs[3].plot(timeAxis, modulation[target, 3, :])

            fig, axs = plt.subplots(4, sharex=True, sharey=True)
            fig.suptitle("Pure, Sample: " + str(target))
            axs[0].plot(timeAxis, modulated[target, 0, :])
            axs[1].plot(timeAxis, modulated[target, 1, :])
            axs[2].plot(timeAxis, modulated[target, 2, :])
            axs[3].plot(timeAxis, modulated[target, 3, :])

            fig, axs = plt.subplots(4, sharex=True, sharey=True)
            fig.suptitle("Real, Sample: " + str(target))
            axs[0].plot(timeAxis, real[target, 0, :])
            axs[1].plot(timeAxis, real[target, 1, :])
            axs[2].plot(timeAxis, real[target, 2, :])
            axs[3].plot(timeAxis, real[target, 3, :])

            fig, axs = plt.subplots(4, sharex=True, sharey=True)
            fig.suptitle("Real and pure, Sample: " + str(target))
            axs[0].plot(timeAxis, real[target, 0, :])
            axs[0].plot(timeAxis, modulated[target, 0, :])
            axs[1].plot(timeAxis, real[target, 1, :])
            axs[1].plot(timeAxis, modulated[target, 1, :])
            axs[2].plot(timeAxis, real[target, 2, :])
            axs[2].plot(timeAxis, modulated[target, 2, :])
            axs[3].plot(timeAxis, real[target, 3, :])
            axs[3].plot(timeAxis, modulated[target, 3, :])

            fig, axs = plt.subplots(4, sharex=True, sharey=True)
            fig.suptitle("Noise, Sample: " + str(target))
            axs[0].plot(timeAxis, noise[target, 0, :])
            axs[1].plot(timeAxis, noise[target, 1, :])
            axs[2].plot(timeAxis, noise[target, 2, :])
            axs[3].plot(timeAxis, noise[target, 3, :])

            fig, axs = plt.subplots(4, sharex=True, sharey=True)
            fig.suptitle("Drift, Sample: " + str(target))
            axs[0].plot(timeAxis, drift[target, 0, :])
            axs[1].plot(timeAxis, drift[target, 1, :])
            axs[2].plot(timeAxis, drift[target, 2, :])
            axs[3].plot(timeAxis, drift[target, 3, :])
            plt.show()

        return
