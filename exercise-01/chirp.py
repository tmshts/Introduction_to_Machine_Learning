import matplotlib.pyplot as plt
import numpy as np

def createChirpSignal(samplingrate:int, duration:int, freqfrom:int, freqto:int, linear: bool=True):
    # returns the chirp signal as list or 1D-array
    if linear == True:
        c = (freqto - freqfrom) / duration
        # linear space
        t = np.linspace(0, duration, samplingrate)
        # formula for linear frequency chirp
        chirp = np.sin(2 * np.pi * ((c * t**2) / 2 + (freqfrom * t)))
        #plt.plot(t, chirp)
        #plt.grid(True, which="both")
        #plt.title("Linear Frequency Chirp")
        #plt.xlabel('t (sec)')
        #plt.show()
    else: # linear == False:
        k = (freqto / freqfrom)**(1/duration)
        # exponential space
        t = np.linspace(0, duration, samplingrate) # np.logspace
        # formula for exponential frequency chirp
        chirp = np.sin(2 * np.pi * freqfrom * (((k**t) - 1)/np.log(k)))
        #plt.plot(t, chirp)
        #plt.title("Exponential Frequency Chirp")
        #plt.xlabel('t (sec)')
        #plt.show()
    return chirp