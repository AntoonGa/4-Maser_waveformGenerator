"""
This repo contains the 4-maser waveform generator used during my Axion Dark-Matter research project with USTC and JGU-mainz.
In short, the lab at USTC had 4 Xenon-MASER outputting pure sinewaves for weeks on, which could be modulated (frequency/Amplitude) by Axions.
The point of this engine is to generate large numbers of accurate simulations of this setup.
We would then go on an train our models on synthetic data, to later perform inference on the lab data (all of which must be analysed, preventing us from using lab-data to train).

This engine is rather flexible and provide highly accurate signals, which can be modified in the following ways:
Frequency modulation
Amplitude modulation
Multiple sinewve interference
Gaussian (multiple) noise, drift, power line noise augmented
Spurious event (e.g door slam in the lab)
4 channels can be used together or separatly
Lag can be added to individual channel, simulating an event propagating in the lab. Increasing lag is equivalent to physically moving the setups in space, or accelerating the event propagation velocity.
"""