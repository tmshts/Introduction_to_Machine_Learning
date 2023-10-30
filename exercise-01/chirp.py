import numpy as np
from matplotlib import pyplot as plt

def createChirpSignal(samplingrate: int, duration: int, freqfrom: int, freqto: int, linear: bool):
    # returns the chirp signal as list or 1D-array
    # TODO
    if linear:
        t = np.linspace(0, duration, samplingrate)
        k = (freqto - freqfrom) / duration
        y = np.sin(2 * np.pi * (freqfrom + k/2 * t) * t)
        #integral of linear phase from 0 to t
    else:
        if (freqfrom == 0) or (freqto == 0) or (freqto * freqfrom < 0):
            return None
        else:
            t = np.linspace(0, duration, samplingrate)
            k = (freqto / freqfrom) ** (1 / duration)
            y = np.sin(2 * np.pi * freqfrom / np.log(k) * (np.power(k, t) - 1))
            #integral of exponential phase from 0 to t
    plt.plot(t, y)
    plt.ylabel('Amplitude')
    plt.xlabel('Time')
    plt.title('Chirp Signal')
    plt.show()
    return y
