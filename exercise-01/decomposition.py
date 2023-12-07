import matplotlib.pyplot as plt
import numpy as np
from math import pi

# samples = 200, frequency = 2, kMax = 10000, [amplitude = 1])
def createTriangleSignal(samples: int, frequency: int, k_max: int):
    # returns the signal as 1D-array (np.ndarray)
    # linspace(start, stop, number)
    t = np.linspace(0, frequency, num=samples)
    trian = 0
    for k in range(0, k_max - 1):
        trian += (8 / (pi ** 2)) * ((-1)**k) * ((np.sin(2*pi*(2*k + 1)* t))/((2*k + 1)**2))
    #plt.plot(t, trian)
    #plt.grid(True, which="both")
    #plt.title("Triangle Signal")
    #plt.xlabel('Time t (sec)')
    #plt.ylabel('Amplitude')
    #plt.show()
    return trian

def createSquareSignal(samples: int, frequency: int, k_max: int):
    # returns the signal as 1D-array (np.ndarray)
    t = np.linspace(0, frequency, num=samples)
    squa = 0
    for k in range(1, k_max):
        squa += (4 / pi) * ((np.sin(2*pi*(2*k - 1)*t))/((2*k) - 1))
    #plt.plot(t, squa)
    #plt.grid(True, which="both")
    #plt.title("Square Signal")
    #plt.xlabel('Time t (sec)')
    #plt.ylabel('Amplitude')
    #plt.show()
    return squa

def createSawtoothSignal(samples: int, frequency: int, k_max: int, amplitude: int):
    # returns the signal as 1D-array (np.ndarray)
    t = np.linspace(0, frequency, num=samples)
    saw = (amplitude/2)
    for k in range(1, k_max + 1):
        saw -= (amplitude/pi) * (np.sin(2*pi*k*t)/k)
    #plt.plot(t, saw)
    #plt.grid(True, which="both")
    #plt.title("Saw tooth Signal")
    #plt.xlabel('Time t (sec)')
    #plt.ylabel('Amplitude = sin(time)')
    #plt.show()
    return saw
