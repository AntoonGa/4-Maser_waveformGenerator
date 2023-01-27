"""
About the project: 
This repo contains the 4-maser waveform generator used during my Axion Dark-Matter research project with USTC and JGU-mainz.
In short, the lab at USTC had 4 Xenon-MASER outputting pure sinewaves for weeks on, which could be modulated (frequency/Amplitude) by Axions.
The point of this engine is to generate large numbers of accurate simulations of this setup.
We would then go on an train our models on synthetic data, to later perform inference on the lab data (all of which must be analysed, preventing us from using lab-data to train).

About the engine: 
This engine is rather flexible and provide highly accurate signals, which can be modified in the following ways:
Frequency modulation
Amplitude modulation
Multiple sinewve interference
Gaussian (multiple) noise, drift, power line noise augmented
Spurious events (e.g door slam in the lab)
4 channels can be used together or separatly
Lag can be added to individual channel, simulating an event propagating in the lab. Increasing lag is equivalent to physically moving the setups in space, or accelerating the event propagation velocity.
Because hundreds of Gig are generated from the Engine, across multiples files normalization is hardcoded !!! This avoid serious ML training traps. Using the "worst" paramaters data should always sit in the 0.3-0.7 range. Spurious event can make it go up to 0-1.

Repo files:
The engine class in fourmaser_V5.py
The test file in test.py
The PSD_v1.py is just a quick and dirty script to output normalized PSDs.
"""

![ezcv logo](https://journals.plos.org/plosone/article/figure/image?size=large&id=10.1371/journal.pone.0268439.g001)
Output example. Taken from my paper: "Deep neural networks to recover unknown physical parameters from oscillating time series" (https://doi.org/10.1371/journal.pone.0268439)